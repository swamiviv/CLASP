import os
import sys
import pathlib

sys.path.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), 'pixel2style2pixel'))
from argparse import Namespace
import time
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data

from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from tqdm import tqdm

# psp imports
from datasets import augmentations
from dataset_utils import ImagesDataset, RGBSegMaskDataset, RGBSuperResDataset, RGBSuperResGeneratedDataset
from configs import transforms_config
from models.psp import pSp, get_keys
from models.encoders import psp_encoders

from typing import Dict, Any, List
from dataclasses import dataclass, field
import bounds

import all_utils

from copy import deepcopy

import pandas as pd
import seaborn as sns
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

# Use palplot and pass in the variable:
sns.set()



device = 'cuda'
_DISENT_DATA = [
    [12, 479], [12, 266], [11, 286], [6, 500], [8, 128], [5, 92], [6, 394], [6, 323], [3, 259], [6, 285],
    [5, 414], [6, 128], [9, 295], [6, 322], [6, 487], [6, 504], [6, 497], [6, 501], [15, 45],
    [12, 237], [9, 421], [9, 132], [8, 81], [3, 288], [2, 175], [3, 120], [2, 97], [9, 441],
    [8, 292], [11, 358], [6, 223], [5, 200], [9, 6]
]
_FFHQ_DISENT_DIMS=np.array([6623, 6410, 5918, 3572, 4224, 2652, 3466, 3395, 1795, 3357, 2974,
                                     3200, 4903, 3394, 3559, 3576, 3569, 3573, 7725, 6381, 5029, 4740,
                                     4177, 1824, 1199, 1656, 1121, 5049, 4388, 5990, 3295, 2760, 4614])

_CLEVR_DISENT_DIMS=np.array([7018, 7279, 4557, 3162, 6173, 6263, 2835, 1817, 6220, 6230, 6072,
                             4695, 3456, 3103, 4932, 4311, 4179, 3066, 6044, 4926, 2728, 3144,
                             4512, 7328, 4919, 3071, 5020, 2667])

_STYLE_DIMS= np.asarray([
            0, 512, 512, 512, 512, 512, 512, 512, 512, 512,
            512, 512, 512, 512, 512, 512, 256, 256, 256, 128,
            128, 128, 64, 64, 64, 32, 32,
        ])

@dataclass
class RCPS_Results_Runner:
    base_dir: str
    exp_name: str
    model_name: str
    resize_factors: [int]
    difficulty_levels: [str]
    norm_scheme: str = 'mean'
    mean_style_vector: np.ndarray = field(default_factory=lambda: [])
    style_dims: np.ndarray = field(default_factory=lambda: [])
    results_dict: Dict = field(default_factory=lambda: {})
    device: str = 'cuda:0'

    def __post_init__(self):

        assert len(self.resize_factors) == len(self.difficulty_levels)
        
        self.base_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), self.base_dir)

        self.difficulty_to_resize_factor = {f'{grade}': self.resize_factors[idx] for idx, grade in enumerate(self.difficulty_levels)}

        # to ensure lambda=1 is computed
        self.lambda_values = np.linspace(0, 1, 10, endpoint=False)
        self.lambda_values = np.append(self.lambda_values, np.linspace(1, 10, 100))

        self.all_prediction_sets = {f'{sf}': [] for sf in self.difficulty_levels}
        self.all_losses_per_lambda = {f'{sf}': [] for sf in self.difficulty_levels}
        self.all_rcps_stats = {f'{sf}': {} for sf in self.difficulty_levels}
        self.test_idx_list = {f'{sf}': [] for sf in self.difficulty_levels}

        self.disent_dims = _FFHQ_DISENT_DIMS

        self.select_indices = range(len(self.disent_dims))
        self.load_data()
        self.encoder_net, self.pretrained_model = self.load_encoder_models()


    @property
    def ckpt_path(self):
        return os.path.join(self.base_dir, self.exp_name, self.model_name)

    @property
    def calibration_results_dir(self):
        return os.path.join(self.base_dir, self.exp_name, 'data/calibration_set_outputs_generated_data') 
    
    @property
    def pretrained_models_dir(self):
        return os.path.join(self.base_dir, self.exp_name, 'pretrained_models')

    def get_calibration_file_by_difficulty_level(self, difficulty_level):
        return os.path.join(self.calibration_results_dir, f'{self.difficulty_to_resize_factor[difficulty_level]}', 'outputs.npz')

    def get_coverage_fraction(self, gt, lower_q, upper_q):
        greater_than_lower_q = (gt > lower_q)
        lesser_than_upper_q = (gt < upper_q)
        common_mask = greater_than_lower_q * lesser_than_upper_q
        return common_mask

    def stylespace_flat_to_list(self, spv_flat, style_dims):
        if style_dims[0] != 0:
            style_dims = np.insert(style_dims, 0, 0)
        style_dims = np.cumsum(style_dims)
        spv_list = []
        for idx in range(len(style_dims) - 1):
            start_dim = style_dims[idx]
            end_dim = style_dims[idx + 1]
            spv_list.append(torch.Tensor(spv_flat[:, start_dim:end_dim][:, None, :]).to(device))
        return spv_list

    def get_fnr(self, prediction_set, y):
        lower_edge = prediction_set[0].squeeze()
        upper_edge = prediction_set[1].squeeze()
        assert lower_edge.shape == y.shape
        assert upper_edge.shape == y.shape
        coverage_fractions = self.get_coverage_fraction(y, lower_edge, upper_edge)
        return 1. - coverage_fractions

    def get_image_from_style_vectors(self, spv_flat):
        spv_list = self.stylespace_flat_to_list(spv_flat[None, :], style_dims=_STYLE_DIMS)
        im, _ = self.pretrained_model.decoder.forward_with_style_vectors(
            latent_codes=torch.randn(1, 512), style_vectors=spv_list, input_is_latent=True, randomize_noise=False,
            return_latents=True, modulate=False
        )
        return im

    def tensor2im(self, t):
        t = t.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        t = (t + 1) * 127.5
        t[t > 255.] = 255.
        t[t < 0] = 0.
        im_pil = np.array(t).astype(np.uint8)

        return np.asarray(Image.fromarray(im_pil).resize((256, 256), Image.LANCZOS))

    def normalize_diff_image(self, t):
        t = t.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        t = (t + 1) * 0.5
        t[t < 0] = 0.
        t[t > 1] = 1.
        im_pil = np.array(t * 255.).astype(np.uint8)
        im_pil = np.concatenate((im_pil[:, :, 0:1], im_pil[:, :, 0:1], im_pil[:, :, 0:1]), axis=2)
        return np.asarray(Image.fromarray(im_pil).resize((256, 256), Image.LANCZOS))

    def min_max_norm(self, latent, min_latent=None, max_latent=None):
        if min_latent is None:
            min_latent = self.min_style_vector.squeeze()
            max_latent = self.max_style_vector.squeeze()
        assert latent.ndim == min_latent.ndim
        norm_latent = (latent - min_latent) / (max_latent - min_latent)

        return (norm_latent - 0.5) * 2.

    def quantile_regression_l1_nested_sets_from_output(self, lower_preds, upper_preds, predictions, lams):
        lower_edges = []
        upper_edges = []
        for lam in lams:
            lower_diff = predictions - lower_preds
            lower_edge = predictions - lam * (np.maximum(lower_diff, 0.))

            upper_diff = upper_preds - predictions
            upper_edge = lam * (np.maximum(upper_diff, 0.)) + predictions

            if lower_edge.ndim == 3:
                lower_edge = lower_edge[None, :, :, :]
                upper_edge = upper_edge[None, :, :, :]

            lower_edges.append(lower_edge)
            upper_edges.append(upper_edge)
        return lower_edges, upper_edges

    def splice_latent_code(self, latent_code, splice_dims, target_vector):
        spliced_latent_code = deepcopy(latent_code)
        spliced_latent_code[splice_dims] = target_vector
        return spliced_latent_code

    def decode_single_image(self, im):
        predictions = self.encoder_net.forward(im)
        predicted_spv_lq = predictions[:, 0, :].squeeze()
        predicted_spv_uq = predictions[:, 2, :].squeeze()
        predicted_spv = predictions[:, 1, :].squeeze()

        print(predicted_spv.shape, self.mean_style_vector.shape)

        style_vectors = {
            'true_sv': predicted_spv.detach().cpu().numpy() + self.mean_style_vector,
            'predicted_sv': predicted_spv.detach().cpu().numpy() + self.mean_style_vector,
            'predicted_sv_lq': predicted_spv_lq.detach().cpu().numpy() + self.mean_style_vector,
            'predicted_sv_uq': predicted_spv_uq.detach().cpu().numpy() + self.mean_style_vector,
        }

        fig1 = self.show_outputs_across_resize_factors(resize_factor='1', run_index=-1, style_vectors = style_vectors)
        fig2 = self.show_outputs_across_resize_factors(resize_factor='32', run_index=-1, style_vectors=style_vectors)

        return fig1, fig2

    def resize_image_by_factor(self, img_tensor_lr, factor):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[factor]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        im_np = (img_tensor_lr.detach().cpu().numpy().transpose((1, 2, 0)))
        im_np = (im_np + 1) * 0.5
        im_np = im_np.clip(0, 1) * 255.

        return transform(Image.fromarray(im_np.astype(np.uint8)))

    def unnormalize_sv(self, latent_sv, min_sv, max_sv):
        return (latent_sv * (max_sv - min_sv)[None, :, :]) + min_sv[None, :, :]

    def get_modified_latent_codes(self, pointwise_prediction, lq_prediction, uq_prediction, prediction_set_lq=None,
                                  prediction_set_uq=None, disent_dims=None, select_indices=None):

        spliced_sample_latent_code_uq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                           uq_prediction[disent_dims[select_indices]])
        spliced_sample_latent_code_lq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                           lq_prediction[disent_dims[select_indices]])

        if prediction_set_lq is None:
            calib_spliced_sample_latent_code_uq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                                     uq_prediction[disent_dims[select_indices]])
            calib_spliced_sample_latent_code_lq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                                     lq_prediction[disent_dims[select_indices]])
        else:
            calib_spliced_sample_latent_code_uq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                                     prediction_set_uq[select_indices])
            calib_spliced_sample_latent_code_lq = self.splice_latent_code(pointwise_prediction, disent_dims[select_indices],
                                                                     prediction_set_lq[select_indices])

        return spliced_sample_latent_code_uq, spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_uq

    def load_data(self):
        for ssf in self.difficulty_levels:
            outfile = self.get_calibration_file_by_difficulty_level(ssf)
            results_dict_per_sf = {
                f'{ssf}': {
                    'allz_np': 0,
                    'allz_hat_np': 0,
                    'allz_hat_lq_np': 0,
                    'allz_hat_uq_np': 0,
                }
            }
            with np.load(outfile) as data:
                self.mean_style_vector = data['mean_style_space_vector']
                self.min_style_vector = data['min_style_space_vector']
                self.max_style_vector = data['max_style_space_vector']
                self.style_dims = data['style_dims']
                results_dict_per_sf[f'{ssf}']['allz_np'] = np.vstack(data['all_sv']).squeeze()

                if self.norm_scheme=='mean':
                    results_dict_per_sf[f'{ssf}']['allz_hat_np'] = np.vstack(data['all_sv_hat']) + self.mean_style_vector
                    results_dict_per_sf[f'{ssf}']['allz_hat_lq_np'] = np.vstack(
                        data['all_sv_hat_lq']) + self.mean_style_vector
                    results_dict_per_sf[f'{ssf}']['allz_hat_uq_np'] = np.vstack(
                        data['all_sv_hat_uq']) + self.mean_style_vector

                elif self.norm_scheme=='min_max':
                    results_dict_per_sf[f'{ssf}']['allz_hat_np'] = self.min_max_norm(
                        np.vstack(data['all_sv_hat']), data['min_style_space_vector'], data['max_style_space_vector']
                    )
                    results_dict_per_sf[f'{ssf}']['allz_hat_lq_np'] = self.min_max_norm(
                        np.vstack(data['all_sv_hat_lq']), data['min_style_space_vector'], data['max_style_space_vector']
                    )
                    results_dict_per_sf[f'{ssf}']['allz_hat_uq_np'] = self.min_max_norm(
                        np.vstack(data['all_sv_hat_uq']), data['min_style_space_vector'], data['max_style_space_vector']
                    )

                else:
                    results_dict_per_sf[f'{ssf}']['allz_hat_np'] = np.vstack(data['all_sv_hat'])
                    results_dict_per_sf[f'{ssf}']['allz_hat_lq_np'] = np.vstack(data['all_sv_hat_lq'])
                    results_dict_per_sf[f'{ssf}']['allz_hat_uq_np'] = np.vstack(data['all_sv_hat_uq'])

            self.results_dict.update(results_dict_per_sf)

    def compute_losses_prediction_sets(self):
        for sf in self.difficulty_levels:
            ssf_stats = self.results_dict[f'{sf}']
            allz_np = ssf_stats['allz_np']
            allz_hat_np = ssf_stats['allz_hat_np']
            allz_hat_lq_np = ssf_stats['allz_hat_lq_np']
            allz_hat_uq_np = ssf_stats['allz_hat_uq_np']
            prediction_sets = self.quantile_regression_l1_nested_sets_from_output(
                allz_hat_lq_np[:, self.disent_dims[self.select_indices]], allz_hat_uq_np[:, self.disent_dims[self.select_indices]],
                allz_hat_np[:, self.disent_dims[self.select_indices]], self.lambda_values
            )

            loss_value_per_lambda = []
            for idx in range(len(self.lambda_values)):
                loss_value_per_lambda.append(self.get_fnr((prediction_sets[0][idx], prediction_sets[1][idx]),
                                                     allz_np.squeeze()[:, self.disent_dims[self.select_indices]]))

            self.all_prediction_sets[f'{sf}'].append(prediction_sets)
            self.all_losses_per_lambda[f'{sf}'].append(loss_value_per_lambda)


    def calibrate_per_difficulty_level(self, resize_factor, total_runs=50):
        np_rng = np.random.RandomState(2021)
        seeds = np_rng.randint(0, 1000, total_runs)
        L = self.all_losses_per_lambda['1'][0][0].shape[0]

        val_idx_list = []
        test_idx_list = []

        for run_id in range(total_runs):
            np.random.seed(seeds[run_id])
            val_idx_list.append(np.random.choice(np.arange(0, L), size=int(0.8 * L), replace=False))
            test_idx_list.append(np.setdiff1d(np.arange(0, L), val_idx_list[-1]))
        self.test_idx_list[f'{resize_factor}'] = test_idx_list

        for k, v in self.all_losses_per_lambda.items():
            if k != f'{resize_factor}':
                continue
            all_losses = [loss_matrix.mean(axis=-1) for loss_matrix in v[0]]
            all_losses = np.vstack(all_losses)
            prediction_sets = self.all_prediction_sets[k][0]
            print(all_losses.mean(), all_losses.shape, len(prediction_sets[0]))
            test_set_sizes, emp_risks, idx_lambda_calib_all_runs = all_utils.get_rcps_stats(
                val_idx_list, test_idx_list, self.lambda_values, total_runs, all_losses, prediction_sets[0], prediction_sets[1],
            )
            rcps_stats = {'test_set_sizes': test_set_sizes, 'emp_risks': emp_risks, 'lambda_calib': self.lambda_values[idx_lambda_calib_all_runs],
                          'idx_lambda_calib': idx_lambda_calib_all_runs}
            self.all_rcps_stats[k].update(rcps_stats)

    def calibrate_all_difficulty_levels(self, total_runs=50):
        np_rng = np.random.RandomState(2021)
        seeds = np_rng.randint(0, 1000, total_runs)

        all_losses = []
        all_lower_edges = []
        all_upper_edges = []
        all_emp_risks = []
        self.indices_per_difficulty_level = {f'{grade}': [] for grade in self.difficulty_levels}
        running_start_index = 0
        for grade in self.difficulty_levels:
            self.indices_per_difficulty_level[grade] = range(running_start_index, running_start_index + len(
                self.all_losses_per_lambda[grade][0][0]))
            running_start_index += len(self.all_losses_per_lambda[grade][0][0])

        for lambda_index, lambda_value in enumerate(self.lambda_values):
            loss_per_lambda = []
            lower_edges_per_lambda = []
            upper_edges_per_lambda = []
            for difficulty_level in self.difficulty_levels:
                loss_per_lambda.append(self.all_losses_per_lambda[difficulty_level][0][lambda_index])
                lower_edges_per_lambda.append(self.all_prediction_sets[difficulty_level][0][0][lambda_index])
                upper_edges_per_lambda.append(self.all_prediction_sets[difficulty_level][0][1][lambda_index])
            all_losses.append(np.concatenate(loss_per_lambda, axis=0).mean(axis=-1)[None, :])
            all_lower_edges.append(np.concatenate(lower_edges_per_lambda, axis=0)[None, :])
            all_upper_edges.append(np.concatenate(upper_edges_per_lambda, axis=0)[None, :])
        all_losses = np.concatenate(all_losses, axis=0)
        all_lower_edges = np.concatenate(all_lower_edges, axis=0)
        all_upper_edges = np.concatenate(all_upper_edges, axis=0)

        L = len(self.indices_per_difficulty_level[self.difficulty_levels[0]])


        val_idx_list = []
        test_idx_list = []
        for run_id in range(total_runs):
            np.random.seed(seeds[run_id])
            val_idx_list.append(np.random.choice(np.arange(0, L), size=int(0.8 * L), replace=False))
            test_idx_list.append(np.setdiff1d(np.arange(0, L), val_idx_list[-1]))
        self.test_idx_list[self.difficulty_levels[0]] = test_idx_list

        all_set_sizes, all_emp_risks, mean_emp_risks, idx_lambda_calib_all_runs = all_utils.get_rcps_stats(
            val_idx_list, test_idx_list, self.indices_per_difficulty_level, self.lambda_values, total_runs, all_losses, all_lower_edges, all_upper_edges,
        )
        rcps_stats = {'all_set_sizes': all_set_sizes, 'all_emp_risks': all_emp_risks, 'mean_emp_risks': mean_emp_risks,
                      'lambda_calib': self.lambda_values[idx_lambda_calib_all_runs],
                      'idx_lambda_calib': idx_lambda_calib_all_runs}

        # Allocate back to different difficulty levels
        for difficulty_level in self.difficulty_levels:
            # Init
            self.all_rcps_stats[difficulty_level].update(rcps_stats)
            # Change test set size index to only samples from difficulty level
            set_sizes_per_difficulty_level = []
            emp_risks_per_difficulty_level = []
            for emp_risks, set_size in zip(all_emp_risks, all_set_sizes):
                set_sizes_per_difficulty_level.append(set_size[self.indices_per_difficulty_level[difficulty_level], :])
                emp_risks_per_difficulty_level.append(emp_risks[self.indices_per_difficulty_level[difficulty_level]])
            self.all_rcps_stats[difficulty_level]['all_set_sizes'] = set_sizes_per_difficulty_level
            self.all_rcps_stats[difficulty_level]['all_emp_risks'] = emp_risks_per_difficulty_level

    def visualize_style_space_intervals(self, difficulty_level, image_index, plot_type='calibration_interval', run_index=-1, select_groups=None, group_names=None):

        assert plot_type in ['true_value', 'predicted_value', 'prediction_interval', 'calibration_interval']
        SMALL_SIZE = 8
        BIGGER_SIZE = 12

        plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
        plt.rc('legend', fontsize=20)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # Manual mapping specification based on Style Space Analysis paper
        interp_dim_groups = {
            'name': ['hc0', 'hc1', 'hc2', 'hl3', 'hl4', 'hl5', 'hl6', 'hl7', 'hs8', 'hs9', 'hs10', 'hs11', 'hs12', 'hs13', 'hs14', 'hs15', 'hs16', 'm17', 'm18', 'e19', 'e20', 'e21',
                     'e22', 'g23'],
            'plot_dims': range(24),
            'desc': ['hc0', 'hc1', 'hc2', 'hl3', 'hl4', 'hl5', 'hl6', 'hl7', 'hs8', 'hs9', 'hs10', 'hs11', 'hs12', 'hs13', 'hs14', 'hs15', 'hs16', 'm17', 'm18', 'e19', 'e20', 'e21',
                     'e22', 'g23'],
        }

        plot_dim_indices = interp_dim_groups['plot_dims'] if select_groups is None else np.asarray(interp_dim_groups['plot_dims'])[select_groups]
        att_names = interp_dim_groups['desc'] if select_groups is None else group_names

        # plot_dim_indices = range(33)
        gt_values = []
        point_predictions = []
        calib_intervals_lq = []
        calib_intervals_uq = []

        predicted_intervals_lq = []
        predicted_intervals_uq = []

        resize_factor = self.difficulty_to_resize_factor[difficulty_level]
        lr_stats = self.results_dict[difficulty_level]
        lr_prediction_sets = self.all_prediction_sets[difficulty_level][0]
        prediction_sets = (lr_prediction_sets[0][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]],
                           lr_prediction_sets[1][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]])
        # sample_index = test_idx_list[-1][interesting_indices[0]]
        sample_latent_code_prediction = lr_stats['allz_hat_np'][image_index]
        sample_latent_code_lq = lr_stats['allz_hat_lq_np'][image_index]
        sample_latent_code_uq = lr_stats['allz_hat_uq_np'][image_index]
        sample_true_latent_code = lr_stats['allz_np'][image_index]
        spliced_sample_latent_code_uq, spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_uq = self.get_modified_latent_codes(
            sample_latent_code_prediction, sample_latent_code_lq, sample_latent_code_uq,
            prediction_set_lq=prediction_sets[0][image_index], prediction_set_uq=prediction_sets[1][image_index],
            disent_dims=self.disent_dims, select_indices=self.select_indices
        )

        sample_true_latent_code = self.min_max_norm(sample_true_latent_code)
        sample_latent_code_prediction = self.min_max_norm(sample_latent_code_prediction)
        calib_spliced_sample_latent_code_lq = self.min_max_norm(calib_spliced_sample_latent_code_lq)
        calib_spliced_sample_latent_code_uq = self.min_max_norm(calib_spliced_sample_latent_code_uq)
        spliced_sample_latent_code_lq = self.min_max_norm(spliced_sample_latent_code_lq)
        spliced_sample_latent_code_uq = self.min_max_norm(spliced_sample_latent_code_uq)


        for idx in plot_dim_indices:
            gt_values.extend([sample_true_latent_code[self.disent_dims[idx]]])
            point_predictions.extend([sample_latent_code_prediction[self.disent_dims[idx]]])
            calib_intervals_lq.extend([calib_spliced_sample_latent_code_lq[self.disent_dims[idx]]])
            calib_intervals_uq.extend([calib_spliced_sample_latent_code_uq[self.disent_dims[idx]]])
            predicted_intervals_lq.extend([spliced_sample_latent_code_lq[self.disent_dims[idx]]])
            predicted_intervals_uq.extend([spliced_sample_latent_code_uq[self.disent_dims[idx]]])

        # Check where calibration helps
        predicted_coverage = [(predicted_intervals_lq[idx] < gt_values[idx] < predicted_intervals_uq[idx]) for idx in range(len(gt_values))]
        calibrated_coverage = [(calib_intervals_lq[idx] < gt_values[idx] < calib_intervals_uq[idx]) for idx in range(len(gt_values))]
        num_dims_covered = [(~predicted_coverage[idx] & calibrated_coverage[idx]) for idx in range(len(predicted_coverage))]
        num_dims_covered = np.asarray(num_dims_covered).sum()


        fig = plt.figure(figsize=(24, 12))
        plt.plot(np.asarray(gt_values)+0.05, range(len(gt_values)), marker="o", markerfacecolor="lime", label='True value', linestyle="None", markersize=12)
        plt.xlim([-1., 1.])
        xlimits = plt.gca().get_xlim()
        xlength = xlimits[1] - xlimits[0]

        def _plot_shaded_interval(lq, uq, xlimits, xlength, yval, label=None, color=None):
            lq_rel_value = (lq - xlimits[0]) / xlength
            uq_rel_value = (uq - xlimits[0]) / xlength
            plt.gca().axhspan(yval-0.025, yval+0.025,  xmin=lq_rel_value, xmax=uq_rel_value, alpha=0.6, color=color or 'cyan', label=label)

        if plot_type != 'true_value':
            plt.plot(point_predictions, range(len(point_predictions)), marker="o", markerfacecolor="blue", markersize=12, label='Point prediction', linestyle="None")


        if plot_type == 'prediction_interval' or plot_type == 'calibration_interval':
            for attribute_num in range(len(plot_dim_indices)):
                # TODO: use consistent indexing into predicted_intervals
                label = None
                if attribute_num == len(plot_dim_indices) - 1:
                    label = 'Predicted interval'
                _plot_shaded_interval(predicted_intervals_lq[attribute_num], predicted_intervals_uq[attribute_num], xlimits, xlength, attribute_num - 0.03, label)

        if plot_type == 'calibration_interval':
            for attribute_num in range(len(plot_dim_indices)):
                # TODO: use consistent indexing into predicted_intervals
                label = None
                if attribute_num == len(plot_dim_indices) - 1:
                    label = 'Calibrated interval'
                _plot_shaded_interval(calib_intervals_lq[attribute_num], calib_intervals_uq[attribute_num], xlimits, xlength, attribute_num + 0.03, label=label, color='red')

        # get the positions for the maximum ytick label
        #y_max = int(max(plt.yticks()[0]))  # int() to convert numpy.int32 => int
        # manually set you ytick labels
        plt.yticks(np.arange(len(att_names)), att_names)
        plt.xticks([-1.0, -0.5, 0, 0.5, 1.0])
        plt.ylabel('Semantic Factor', fontsize=14)
        plt.gca().tick_params(axis='both', which='major', labelsize=12)
        plt.gca().tick_params(axis='both', which='minor', labelsize=10)
        #plt.grid()
        leg = plt.legend(loc='best', prop={'size': 12})
        for lh in leg.legendHandles:
            lh.set_alpha(0.6)


        return fig, num_dims_covered

    def plot_set_sizes(self):
        desc_strs = []
        fig = plt.figure(figsize=(20, 10))
        bins = np.linspace(0, 10, 100)
        colors = ['k', 'r', 'g', 'b', 'c']

        for idx, ssf in enumerate(self.difficulty_levels):
            ssf_stats = self.results_dict[f'{ssf}']
            set_sizes = np.abs(ssf_stats['allz_hat_lq_np'] - ssf_stats['allz_hat_uq_np'])
            plt.hist(set_sizes[:, self.disent_dims[self.select_indices]].ravel(), bins=bins, alpha=0.5)
            desc_strs.append('{}'.format(ssf))

        plt.legend(desc_strs, fontsize=14)
        plt.gca().axes.yaxis.set_ticklabels([])

        all_utils.decorate_plot(ax=plt.gca(), xlabel='Set size', ylabel='Histrogram count', title_str='CELEB-A Super-resolution: Set sizes across difficulty levels')
        return fig

    def plot_residuals(self):

        set_sizes_dict = {
            factor: (self.results_dict[f'{factor}']['allz_np'] - self.results_dict[f'{factor}']['allz_hat_np'])[:,
                    self.disent_dims[self.select_indices]].ravel() for factor in self.difficulty_levels
        }
        res_values = []
        set_size_values = []
        for k, v in set_sizes_dict.items():
            for item in v:
                res_values.append(k)
                set_size_values.append(item)

        ylabel = '(prediction - true-value)'
        xlabel = 'Difficulty level'
        dataset = {
            xlabel: res_values,
            ylabel: set_size_values,
        }
        ax = sns.boxplot(x=xlabel, y=ylabel, data=pd.DataFrame(dataset))
        ax.yaxis.set_ticklabels([])
        ax.set(xticklabels=[])

        all_utils.decorate_plot(ax=ax, xlabel='Difficulty level', ylabel='(prediction - true-value)', title_str='Residuals across difficulty levels')


        return ax


    def decode_single_image_across_difficulty_levels(self, im, resize_factor, calib_lambda):
        transform_obj = transforms.Compose([transforms.Resize((256, 256)),
        augmentations.BilinearResize(factors=[1], probabilities=[1.0]),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        im_truth_raw_np = deepcopy(im)
        if not torch.is_tensor(im):
            im = transform_obj(Image.fromarray(im.astype(np.uint8))).to(self.device).float().unsqueeze(0)
        im_truth = deepcopy(im)
        if int(resize_factor) > 1:
            transform_obj = transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.BilinearResize(factors=[resize_factor], probabilities=[1.0]),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            im = transform_obj(Image.fromarray(im_truth_raw_np.astype(np.uint8))).to(self.device).float().unsqueeze(0)

        print(f'After resize: {im.min()}, {im.max()}')

        predictions = self.encoder_net.forward(im)

        predictions = predictions.detach().cpu().numpy()
        if self.norm_scheme == 'mean':
            predictions += self.mean_style_vector[None, None, :]
        predicted_spv_lq = predictions[:, 0, :]
        predicted_spv_uq = predictions[:, 2, :]
        predicted_spv = predictions[:, 1, :]

        print(im.sum(), im_truth.sum(), predicted_spv.sum())

        lower_diff = predicted_spv - predicted_spv_lq
        lower_edge = predicted_spv - calib_lambda * (np.maximum(lower_diff, 0.))

        upper_diff = predicted_spv_uq - predicted_spv
        upper_edge = calib_lambda * (np.maximum(upper_diff, 0.)) + predicted_spv
        upper_edge = upper_edge.squeeze()
        lower_edge = lower_edge.squeeze()
        predicted_spv = predicted_spv.squeeze()

        with torch.no_grad():
            im_lq = self.get_image_from_style_vectors(
                self.splice_latent_code(predicted_spv, self.disent_dims[self.select_indices], lower_edge[self.disent_dims[self.select_indices]])
            )
            im_uq = self.get_image_from_style_vectors(
                self.splice_latent_code(predicted_spv, self.disent_dims[self.select_indices], upper_edge[self.disent_dims[self.select_indices]])
            )
            im_prediction = self.get_image_from_style_vectors(predicted_spv)

        im_output_array = np.concatenate(
            (
                self.tensor2im(im),
                self.tensor2im(im_truth),
                self.tensor2im(im_prediction),
                self.tensor2im(im_lq),
                self.tensor2im(im_uq),
            ),
            axis=1
        )

        return im_output_array

    def interpolate_quantiles_along_each_factor(self, difficulty_level, image_index=None, run_index=-1):
        resize_factor = self.difficulty_to_resize_factor[difficulty_level]
        lr_stats = self.results_dict[difficulty_level]
        sample_true_latent_code = self.results_dict[difficulty_level]['allz_np'][image_index]
        sample_latent_code_prediction = lr_stats['allz_hat_np'][image_index]
        sample_latent_code_lq = lr_stats['allz_hat_lq_np'][image_index]
        sample_latent_code_uq = lr_stats['allz_hat_uq_np'][image_index]

        im_y_hat = self.get_image_from_style_vectors(sample_true_latent_code)
        lr_prediction_sets = self.all_prediction_sets[difficulty_level][0]
        prediction_sets = (lr_prediction_sets[0][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]],
                           lr_prediction_sets[1][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]])

        im_x = im_y_hat[0]
        if int(resize_factor) > 1:
            im_x = self.resize_image_by_factor(im_x, int(resize_factor))

        spliced_sample_latent_code_uq, spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_uq = self.get_modified_latent_codes(
            sample_latent_code_prediction, sample_latent_code_lq, sample_latent_code_uq,
            prediction_set_lq=prediction_sets[0][image_index], prediction_set_uq=prediction_sets[1][image_index],
            disent_dims=self.disent_dims, select_indices=self.select_indices
        )

        interpolated_latents = all_utils.interpolate(calib_spliced_sample_latent_code_lq,
                                                     calib_spliced_sample_latent_code_uq,
                                                     weights=torch.Tensor(np.arange(0., 1.0, 0.9)).to(self.device))

        factor_modified_arrays = []
        interp_dim_groups = {
            'hair_color': range(0, 3),
            'hair_bangs': range(3, 8),
            'hair_shape': range(8, 16),
            'Mouth/Smile': [17, 18],
            'Eye/Glasses': [23, 24, 25, 26],
            'Gender': [32],
        }
        for group_name, group_indices in interp_dim_groups.items():

            ims = []
            for interpolated_latent in interpolated_latents:
                cached_latent = deepcopy(sample_latent_code_prediction)
                cached_latent[self.disent_dims[group_indices]] = interpolated_latent[self.disent_dims[group_indices]]
                ims.append(self.get_image_from_style_vectors(cached_latent))
            intermediate_ims_array = np.concatenate([self.tensor2im(im) for im in ims],  axis=1)
            diff_im = self.normalize_diff_image(ims[-1] - ims[0])
            cm = plt.get_cmap('magma')

            # Apply the colormap like a function to any array:
            colored_image = (cm(diff_im[:, :, 0])* 255).astype(np.uint8)
            intermediate_ims_array = np.concatenate((intermediate_ims_array, colored_image[:, :, :3]), axis=1)

            with torch.no_grad():
                im_lq = self.get_image_from_style_vectors(calib_spliced_sample_latent_code_lq)
                im_uq = self.get_image_from_style_vectors(calib_spliced_sample_latent_code_uq)
                im_prediction = self.get_image_from_style_vectors(sample_latent_code_prediction)


            factor_modified_arrays.append(intermediate_ims_array)

        return factor_modified_arrays



    def show_outputs_across_difficulty_levels(self, difficulty_level, run_index=-1, image_index=None, style_vectors=None):
        assert image_index is not None or style_vectors is not None, 'Either specify image index or provide style vectors fp a bespoke image'
        resize_factor = self.difficulty_to_resize_factor[difficulty_level]
        if style_vectors is not None:
            image_index=50
            sample_true_latent_code = style_vectors['true_sv']
            sample_latent_code_prediction = style_vectors['predicted_sv']
            sample_latent_code_lq = style_vectors['predicted_sv_lq']
            sample_latent_code_uq = style_vectors['predicted_sv_uq']

        else:
            lr_stats = self.results_dict[difficulty_level]
            sample_true_latent_code = self.results_dict[difficulty_level]['allz_np'][image_index]
            sample_latent_code_prediction = lr_stats['allz_hat_np'][image_index]
            sample_latent_code_lq = lr_stats['allz_hat_lq_np'][image_index]
            sample_latent_code_uq = lr_stats['allz_hat_uq_np'][image_index]


        im_y_hat = self.get_image_from_style_vectors(sample_true_latent_code)
        fig = plt.figure(figsize=(20, 20))

        lr_prediction_sets = self.all_prediction_sets[difficulty_level][0]
        prediction_sets = (lr_prediction_sets[0][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]],
                             lr_prediction_sets[1][self.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]])


        spliced_sample_latent_code_uq, spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_lq, calib_spliced_sample_latent_code_uq = self.get_modified_latent_codes(
            sample_latent_code_prediction, sample_latent_code_lq, sample_latent_code_uq,
            prediction_set_lq=prediction_sets[0][image_index], prediction_set_uq=prediction_sets[1][image_index],
            disent_dims=self.disent_dims, select_indices=self.select_indices
        )
        with torch.no_grad():
            im_lq = self.get_image_from_style_vectors(spliced_sample_latent_code_lq)
            im_uq = self.get_image_from_style_vectors(spliced_sample_latent_code_uq)
            im_prediction = self.get_image_from_style_vectors(sample_latent_code_prediction)

        im_x = im_y_hat[0]
        if int(resize_factor) > 1:
            im_x = self.resize_image_by_factor(im_x, int(resize_factor))

        with torch.no_grad():
            im_lq = self.get_image_from_style_vectors(calib_spliced_sample_latent_code_lq)
            im_uq = self.get_image_from_style_vectors(calib_spliced_sample_latent_code_uq)
            im_prediction = self.get_image_from_style_vectors(sample_latent_code_prediction)

        im_array = np.concatenate((
            self.tensor2im(im_x),
            self.tensor2im(im_y_hat),
            self.tensor2im(im_prediction),
            self.tensor2im(im_lq),
            self.tensor2im(im_uq),
            self.normalize_diff_image(im_uq - im_lq),
        ), axis=1)

        interpolated_latents = all_utils.interpolate(calib_spliced_sample_latent_code_lq,
                                                     calib_spliced_sample_latent_code_uq,
                                                     weights=torch.Tensor([0.0, 1.0]).to(self.device))

        factor_modified_arrays = []
        interp_dim_groups = {
            'hair_color': range(0, 3),
            'hair_bangs': range(3, 8),
            'hair_shape': range(8, 16),
            'Mouth/Smile': [17, 18],
            'Eye/Glasses': [23, 24, 25, 26],
            'Gender': [32],
        }
        for group_name, group_indices in interp_dim_groups.items():

            ims = []
            for interpolated_latent in interpolated_latents:
                cached_latent = deepcopy(sample_latent_code_prediction)
                cached_latent[self.disent_dims[group_indices]] = interpolated_latent[self.disent_dims[group_indices]]
                ims.append(self.get_image_from_style_vectors(cached_latent))
            intermediate_ims_array = np.concatenate([self.tensor2im(im) for im in ims], axis=1)
            diff_im = self.normalize_diff_image(ims[-1] - ims[0])
            cm = plt.get_cmap('magma')

            # Apply the colormap like a function to any array:
            colored_image = (cm(diff_im[:, :, 0]) * 255).astype(np.uint8)
            intermediate_ims_array = np.concatenate((intermediate_ims_array, colored_image[:, :, :3]), axis=1)

            factor_modified_arrays.append(intermediate_ims_array)



        return im_array, factor_modified_arrays

    def load_encoder_models(self):
        # Load trained model
        model_to_load = self.ckpt_path
        ckpt = torch.load(model_to_load, map_location='cuda:0')
        encoder_opts = ckpt['opts']
        encoder_opts['checkpoint_path'] = model_to_load
        encoder_opts = Namespace(**encoder_opts)
        encoder_net = psp_encoders.SimpleResnetEncoder_with_quantiles(50, 'ir_se', opts=encoder_opts)

        ckpt_filt = {k: v for k, v in ckpt['state_dict'].items()}
        encoder_net.load_state_dict(ckpt_filt, strict=True)
        encoder_net.eval().cuda()

        latent_avg = ckpt['latent_avg'].to(self.device)
        print(f'Loaded quantile encoder model from {model_to_load}')

        encoder_opts.resize_factors = '32'
        transform_obj = transforms_config.SuperResTransforms(encoder_opts).get_transforms()

        psp_configs = {
            'superres': {
                'batch_size': 32,
                'data_path': '/run/user/61513/loopmnt1',  # os.getenv('CELEBA_MASK_PATH'),
                'source_transform': transform_obj['transform_source'],
                'target_transform': transform_obj['transform_gt_train'],
                'pretrained_model_path': os.path.join(self.pretrained_models_dir, 'psp_celebs_super_resolution.pt'),
            }
        }

        EXP_NAME = 'superres'
        exp_config = psp_configs[EXP_NAME]

        # Load pretrained psp model
        ckpt = torch.load(exp_config['pretrained_model_path'], map_location='cuda:0')
        pretrained_opts = ckpt['opts']
        pretrained_opts['checkpoint_path'] = exp_config['pretrained_model_path']
        pretrained_opts['output_size'] = 1024
        pretrained_opts = Namespace(**pretrained_opts)
        pretrained_latent_avg = ckpt['latent_avg'].to('cuda:0')
        pretrained_psp_net = pSp(pretrained_opts).eval().cuda()
        print(f"Loaded pretrained psp generator model from {exp_config['pretrained_model_path']}")

        return encoder_net, pretrained_psp_net