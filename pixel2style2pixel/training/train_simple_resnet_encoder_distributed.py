import os
import numpy as np
from datetime import datetime
import math
import copy

from argparse import Namespace
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from pixel2style2pixel.utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from pixel2style2pixel.datasets.images_dataset import ImagesDataset, RGBSegMaskDataset, RGBSuperResDataset, \
    RGBSuperResGeneratedDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from models.encoders import psp_encoders
from datasets import augmentations

# QR specific imports
from quantile_helpers import quantile_loss_helper
from configs import transforms_config
from tqdm import tqdm

from PIL import Image
from pixel2style2pixel.training.coach import Coach

_STYLE_SPACE_MIN = -10.
_STYLE_SPACE_MAX = 10.
_DISENT_DATA = [
            [12, 479], [12, 266], [11, 286], [6, 500], [8, 128], [5, 92], [6, 394], [6, 323], [3, 259], [6, 285],
            [5, 414], [6, 128], [9, 295], [6, 322], [6, 487], [6, 504], [6, 497], [6, 501], [15, 45],
            [12, 237], [9, 421], [9, 132], [8, 81], [3, 288], [2, 175], [3, 120], [2, 97], [9, 441],
            [8, 292], [11, 358], [6, 223], [5, 200], [9, 6]
        ]

class Distributed_Trainer_direct_quantile_SimpleResnetEncoder(Coach):
    def __init__(self, opts):
        # super().__init__(opts)
        # opts.lpips_lambda = 0
        # opts.id_lambda = 0
        # opts.w_norm_lambda = 0
        # opts.moco_lambda = 0
        self.opts = opts
        self.device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device
        self.opts.n_style_space_vectors = 26

        self.id_loss = None
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        self.quantile_loss = quantile_loss_helper(params={'q_lo': 0.05, 'q_hi': 0.95}, no_mse=not self.opts.mse)
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        if self.opts.use_wandb:
            from pixel2style2pixel.utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize pretrained model to get latents from
        ckpt = torch.load(opts.pretrained_model_path, map_location=self.opts.device)
        pretrained_opts = ckpt['opts']
        pretrained_opts['checkpoint_path'] = self.opts.pretrained_model_path
        if 'learn_in_w' not in pretrained_opts:
            pretrained_opts['learn_in_w'] = False
        if 'output_size' not in pretrained_opts:
            pretrained_opts['output_size'] = self.opts.output_size

        pretrained_opts = Namespace(**pretrained_opts)
        self.pretrained_psp_net = pSp(pretrained_opts)
        self.pretrained_psp_net.eval().to(opts.device)
        self.pretrained_latent_avg = ckpt['latent_avg'].to(opts.device)
        print('Pretrained psp model successfully loaded!')

        # Initialize network
        # compute number of style inputs based on pretrained model
        self.opts.n_styles = int(math.log(pretrained_opts.output_size, 2)) * 2 - 2
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location=self.device)
            self.net = psp_encoders.SimpleResnetEncoder_with_quantiles(50, mode='ir_se', opts=self.opts).to(
                self.device)
            ckpt_filt = {k: v for k, v in ckpt['state_dict'].items()}
            self.net.load_state_dict(ckpt_filt, strict=True)
        else:
            self.net = psp_encoders.SimpleResnetEncoder_with_quantiles(50, mode='ir_se', opts=self.opts).to(
                self.device)
            self.init_weights(self.net)
        self.net.latent_avg = self.pretrained_latent_avg

        self.net.latent_avg = self.pretrained_latent_avg
        self.net.list_of_style_space_dims = np.asarray([
            0, 512, 512, 512, 512, 512, 512, 512, 512, 512,
            512, 512, 512, 512, 512, 512, 256, 256, 256, 128,
            128, 128, 64, 64, 64, 32, 32,
        ])
        self.style_space_dims = np.hstack(self.net.list_of_style_space_dims).sum()
        self.disentangled_dims, self.disentangled_dims_mask = self.get_disentangled_dims()


        mean_sv, min_sv, max_sv = self.pretrained_psp_net.decoder.mean_style_space_latent(latent_dim=512, style_dims=self.style_space_dims)
        self.net.style_space_latent_avg = self.stylespace_flat_to_list(mean_sv)
        self.net.style_space_latent_min = self.stylespace_flat_to_list(min_sv)
        self.net.style_space_latent_max = self.stylespace_flat_to_list(max_sv)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.global_step = 0

        # Init transforms
        if self.opts.experiment_type == 'super_res':
            transform_obj = transforms_config.DirectSuperResTransforms(opts)
            transforms = transform_obj.get_transforms()
            self.transform_source = transforms['transform_source']
            self.transform_target = transforms['transform_gt_train']
            self.transform_source_test = transforms['transform_source_test']
            self.transform_target_test = transforms['transform_test']
        else:
            self.transform_target = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            self.transform_source = transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()])

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        self.pretrained_psp_net.encoder = nn.DataParallel(self.pretrained_psp_net.encoder)
        self.pretrained_psp_net.decoder = nn.DataParallel(self.pretrained_psp_net.decoder)
        self.net = nn.DataParallel(self.net)

        self.quantile_loss = nn.DataParallel(self.quantile_loss)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def parse_and_log_images(self, x, y, y_hat, y_hat_lq, y_hat_uq, title, subscript=None, id_logs=None,
                             display_count=4):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face: lower': common.tensor2im(y_hat_lq[i]),
                'output_face: central': common.tensor2im(y_hat[i]),
                'output_face: upper': common.tensor2im(y_hat_uq[i]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def configure_datasets(self):
        print('Start: Setup dataloaders')
        print(f'Dataset path: {self.opts.train_data_path}')
        if self.opts.experiment_type == 'super_res':
            train_dataset = RGBSuperResGeneratedDataset(
                db_path=self.opts.train_data_path,
                source_transform=self.transform_source,
                target_transform=self.transform_target,
            )
            val_dataset = RGBSuperResDataset(
                db_path=self.opts.val_data_path, mask_dir_prefix='CelebAMask-HQ-mask/val',
                image_dir_prefix='CelebA-HQ-img',
                source_transform=self.transform_source_test,
                target_transform=self.transform_target_test,
            )
        else:
            train_dataset = RGBSegMaskDataset(db_path=self.opts.data_path, mask_dir_prefix='CelebAMask-HQ-mask/train',
                                              image_dir_prefix='CelebA-HQ-img',
                                              source_transform=self.transform_source,
                                              target_transform=self.transform_target,
                                              )
            val_dataset = RGBSegMaskDataset(db_path=self.opts.data_path, mask_dir_prefix='CelebAMask-HQ-mask/val',
                                            image_dir_prefix='CelebA-HQ-img',
                                            source_transform=self.transform_source,
                                            target_transform=self.transform_target,
                                            )

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        return train_dataset, val_dataset

    def configure_optimizers(self):
        params = list(self.net.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def calc_loss(self, true_latents, predicted_latents, gt_images, inputs):
        loss_dict = {}
        id_logs = None
        id_loss = 0.
        lpips_loss = 0.
        w_norm_loss = 0.
        image_mse_loss = 0.
        predicted_spv_lq = predicted_latents[0]
        predicted_spv = predicted_latents[1]
        predicted_spv_uq = predicted_latents[2]
        quantile_loss_mask = torch.ones_like(self.disentangled_dims_mask).unsqueeze(0).repeat(gt_images.shape[0], 1)
        if self.opts.use_quantile_loss_mask:
            quantile_loss_mask = self.disentangled_dims_mask.unsqueeze(0).repeat(gt_images.shape[0], 1)

        if self.opts.id_lambda > 0 or self.opts.lpips_lambda > 0:

            merged_predictions = [
                torch.cat((
                    predicted_spv_lq[idx].squeeze(),
                    predicted_spv[idx].squeeze(),
                    predicted_spv_uq[idx].squeeze(),
                ), axis=0,
                ) for idx in range(len(predicted_spv))
            ]

            yhats, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                latent_codes=torch.randn_like(predicted_spv[0].squeeze()).repeat(3, 1), style_vectors=merged_predictions,
                input_is_latent=True,
                randomize_noise=False, return_latents=True, modulate=False
            )
            yhats = self.face_pool(yhats)
            if self.opts.id_lambda > 0:
                id_loss = self.id_loss.forward_simple(yhats, gt_images.repeat(3, 1, 1, 1))
            else:
                lpips_loss = self.lpips_loss(yhats, gt_images.repeat(3, 1, 1, 1))


            # fake_y_hat, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
            #     latent_codes=torch.randn_like(predicted_spv[0]), style_vectors=predicted_spv, input_is_latent=True,
            #     randomize_noise=False, return_latents=True, modulate=False
            # )
            # fake_y_hat = self.face_pool(fake_y_hat)
            # fake_y_hat_lq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
            #     latent_codes=torch.randn_like(predicted_spv[0]), style_vectors=predicted_spv_lq,
            #     input_is_latent=True,
            #     randomize_noise=False, return_latents=True, modulate=False
            # )
            # fake_y_hat_lq = self.face_pool(fake_y_hat_lq)
            # fake_y_hat_uq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
            #     latent_codes=torch.randn_like(predicted_spv[0]), style_vectors=predicted_spv_uq,
            #     input_is_latent=True,
            #     randomize_noise=False, return_latents=True, modulate=False
            # )
            # fake_y_hat_uq = self.face_pool(fake_y_hat_uq)
        if self.opts.w_norm_lambda > 0:
            if self.opts.start_from_latent_avg:
                latent_mean_diff = torch.cat(predicted_spv, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)
                latent_mean_diff_uq = torch.cat(predicted_spv_uq, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)
                latent_mean_diff_lq = torch.cat(predicted_spv_lq, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)
            else:
                latent_mean_diff = torch.cat(predicted_spv, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)
                latent_mean_diff_uq = torch.cat(predicted_spv_uq, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)
                latent_mean_diff_lq = torch.cat(predicted_spv_lq, axis=-1).squeeze() - torch.cat(
                    self.net.style_space_latent_avg, axis=-1).squeeze().unsqueeze(0)

            w_norm_loss_center = torch.sum(latent_mean_diff.norm(2, dim=-1)) / latent_mean_diff.shape[0]
            w_norm_loss_uq = torch.sum(latent_mean_diff_uq.norm(2, dim=-1)) / latent_mean_diff_uq.shape[0]
            w_norm_loss_lq = torch.sum(latent_mean_diff_lq.norm(2, dim=-1)) / latent_mean_diff_lq.shape[0]

            w_norm_loss = w_norm_loss_center + w_norm_loss_uq + w_norm_loss_lq

        predicted_spv_with_quantiles = torch.cat(
            (
                torch.cat(predicted_spv_lq, axis=-1).squeeze().unsqueeze(1),
                torch.cat(predicted_spv, axis=-1).squeeze().unsqueeze(1),
                torch.cat(predicted_spv_uq, axis=-1).squeeze().unsqueeze(1),
            ),
            axis=1,
        )

        if isinstance(true_latents, list):
            true_latents = torch.cat(true_latents, axis=-1).squeeze()
        quantile_loss_inputs = (
            predicted_spv_with_quantiles, true_latents.squeeze(), quantile_loss_mask,
        )
        quantile_loss, mse_loss = self.quantile_loss(quantile_loss_inputs)
        total_loss = quantile_loss + mse_loss + self.opts.id_lambda * id_loss + self.opts.lpips_lambda * lpips_loss + self.opts.w_norm_lambda * w_norm_loss

        loss_dict['image_mse_loss'] = float(image_mse_loss)
        loss_dict['lpips_loss'] = float(lpips_loss)
        loss_dict['id_loss'] = float(id_loss)
        loss_dict['w_norm_loss'] = float(w_norm_loss)
        loss_dict['quantile_loss'] = float(quantile_loss)
        loss_dict['mse_loss'] = float(mse_loss)
        loss_dict['loss'] = float(total_loss)
        return total_loss, loss_dict, id_logs

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch
            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y = self.face_pool(y)
                true_latents = self.pretrained_psp_net.encoder(x)
                true_latents_with_avg = true_latents + self.pretrained_latent_avg.unsqueeze(0).repeat(
                    true_latents.shape[0], 1, 1)
                _, _, true_style_vectors = self.pretrained_psp_net.decoder(
                    [true_latents_with_avg], input_is_latent=True, randomize_noise=False,
                    return_latents=False,
                )
                predictions = self.net.forward(x)
                predicted_spv_lq = self.stylespace_flat_to_list(predictions[:, 0, :])
                predicted_spv_uq = self.stylespace_flat_to_list(predictions[:, 2, :])
                predicted_spv = self.stylespace_flat_to_list(predictions[:, 1, :])
                predicted_spv_lq = self.add_mean_vector(predicted_spv_lq)
                predicted_spv_uq = self.add_mean_vector(predicted_spv_uq)
                predicted_spv = self.add_mean_vector(predicted_spv)
                loss, cur_loss_dict, id_logs = self.calc_loss(
                    true_latents=true_style_vectors,
                    predicted_latents=(predicted_spv_lq, predicted_spv, predicted_spv_uq),
                    gt_images=y,
                    inputs=x,
                )
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx == 0:
                with torch.no_grad():
                    fake_y_hat, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                        latent_codes=torch.randn_like(true_latents), style_vectors=predicted_spv,
                        input_is_latent=True,
                        randomize_noise=False, return_latents=True, modulate=False
                    )
                    fake_y_hat_lq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                        latent_codes=torch.randn_like(true_latents), style_vectors=self.splice_latent_code(predicted_spv, predicted_spv_lq),
                        input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                    )
                    fake_y_hat_uq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                        latent_codes=torch.randn_like(true_latents), style_vectors=self.splice_latent_code(predicted_spv, predicted_spv_uq),
                        input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                    )
                    fake_y, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                        latent_codes=torch.randn_like(true_latents), style_vectors=true_style_vectors,
                        input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                    )
                    fake_y_hat = self.face_pool(fake_y_hat)
                    fake_y_hat_lq = self.face_pool(fake_y_hat_lq)
                    fake_y_hat_uq = self.face_pool(fake_y_hat_uq)
                    fake_y = self.face_pool(fake_y)

                self.parse_and_log_images(x=x, y=fake_y, y_hat=fake_y_hat, y_hat_lq=fake_y_hat_lq,
                                          y_hat_uq=fake_y_hat_uq,
                                          title='images/test/faces',
                                          subscript='{:04d}'.format(batch_idx), display_count=2)

                # Log images of first batch to wandb
                if self.opts.use_wandb:
                    self.wb_logger.log_quantile_outputs_to_wandb(
                        x=x, y=fake_y, y_hat=fake_y_hat, y_hat_lq=fake_y_hat_lq, y_hat_uq=fake_y_hat_uq, id_logs=None,
                        prefix="test", step=self.global_step, opts=self.opts
                    )

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def stylespace_flat_to_list(self, spv_flat):
        if spv_flat.ndim == 1:
            spv_flat = spv_flat.unsqueeze(0).unsqueeze(0)
        if spv_flat.ndim == 2:
            spv_flat = spv_flat.unsqueeze(1)
        style_dims_cumsum = np.cumsum(self.net.list_of_style_space_dims)
        spv_list = []
        for idx in range(len(style_dims_cumsum) - 1):
            start_dim = style_dims_cumsum[idx]
            end_dim = style_dims_cumsum[idx + 1]
            spv_list.append(spv_flat[:, :, start_dim:end_dim])
        return spv_list

    def add_mean_vector(self, latents):
        norm_latents = [torch.zeros_like(latents[idx]) for idx in range(len(latents))]
        for idx in range(len(latents)):
            norm_latents[idx] = (latents[idx] + self.net.style_space_latent_avg[idx].squeeze(0))
        return norm_latents

    def get_disentangled_dims(self, disent_data=_DISENT_DATA):
        disent_dims = []
        style_dims_cumsum = np.cumsum(self.net.list_of_style_space_dims)
        for datum in disent_data:
            disent_dims.extend([style_dims_cumsum[datum[0]] + datum[1]])
        disent_dims = np.vstack(disent_dims).flatten()
        assert len(disent_dims) == len(disent_data)

        disentangled_dims_mask = np.zeros(style_dims_cumsum[-1])
        for dim in disent_dims:
            print(dim)
            disentangled_dims_mask[dim] = 1.

        return disent_dims, torch.Tensor(disentangled_dims_mask).to(self.device).float()

    def normalize_latents(self, latents, mode='max'):
        if not self.opts.use_style_vector_average:
            return latents
        norm_latents = [torch.zeros_like(latents[idx]) for idx in range(len(latents))]
        for idx in range(len(latents)):
            if mode == 'mean':
                norm_latents[idx] = (latents[idx] - self.net.style_space_latent_avg[idx]) / _STYLE_SPACE_MAX
            if mode == 'max':
                norm_latents[idx] = (latents[idx] - self.net.style_space_latent_min[idx]) / (self.net.style_space_latent_max[idx] - self.net.style_space_latent_min[idx] + 1e-6)

        return norm_latents

    def unnormalize_latents(self, latents, mode='mean'):
        if not self.opts.use_style_vector_average:
            return latents
        unnorm_latents = [torch.zeros_like(latents[idx]) for idx in range(len(latents))]
        for idx in range(len(latents)):
            if mode == 'mean':
                unnorm_latents[idx] = (latents[idx] * _STYLE_SPACE_MAX) + self.net.style_space_latent_avg[idx]
            if mode == 'max':
                unnorm_latents[idx] = (latents[idx] * (self.net.style_space_latent_max[idx] - self.net.style_space_latent_min[idx])) + self.net.style_space_latent_min[idx]

        return unnorm_latents

    def splice_latent_code(self, latent_code_batch, target_vector_batch, splice_dims=None):
        if splice_dims is None:
            splice_dims = _DISENT_DATA
        spliced_latent_code_batch = [latent_code_batch[idx].clone().detach() for idx in range(len(latent_code_batch))]

        for layer, channel in splice_dims:
            if latent_code_batch[0].ndim == 3:
                spliced_latent_code_batch[layer][:, :, channel] = target_vector_batch[layer][:, :, channel]
            else:
                spliced_latent_code_batch[layer][:, channel] = target_vector_batch[layer][:, channel]
        return spliced_latent_code_batch

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                x, y, true_sv, true_w, _ = batch
                x, y, true_sv, true_w = x.to(self.device).float(), y.to(self.device).float(), true_sv.to(self.device).float(), true_w.to(self.device).float()
                y = self.face_pool(y)
                predictions = self.net.forward(x)
                predicted_sv_lq = self.stylespace_flat_to_list(predictions[:, 0, :])
                predicted_sv_uq = self.stylespace_flat_to_list(predictions[:, 2, :])
                predicted_sv = self.stylespace_flat_to_list(predictions[:, 1, :])
                predicted_sv_lq = self.add_mean_vector(predicted_sv_lq)
                predicted_sv_uq = self.add_mean_vector(predicted_sv_uq)
                predicted_sv = self.add_mean_vector(predicted_sv)
                loss, loss_dict, id_logs = self.calc_loss(
                    true_latents=true_sv,
                    predicted_latents=(predicted_sv_lq, predicted_sv, predicted_sv_uq),
                    gt_images=y,
                    inputs=x,
                )
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 100 == 0) or batch_idx == 0:
                    # predicted_latents_with_avg = predicted_latents_with_quantiles[:, 1, :,
                    # 							 :].squeeze() + self.pretrained_latent_avg.unsqueeze(0).repeat(
                    # 	true_latents.shape[0], 1, 1)
                    # predicted_latents_with_avg_lq = predicted_latents_with_quantiles[:, 0, :,
                    # 								:].squeeze() + self.pretrained_latent_avg.unsqueeze(0).repeat(
                    # 	true_latents.shape[0], 1, 1)
                    # predicted_latents_with_avg_uq = predicted_latents_with_quantiles[:, 2, :,
                    # 								:].squeeze() + self.pretrained_latent_avg.unsqueeze(0).repeat(
                    # 	true_latents.shape[0], 1, 1)
                    with torch.no_grad():
                        # predicted_sv_unnorm = self.mean_unnormalize_latents(predicted_sv)
                        # predicted_sv_lq_unnorm = self.mean_unnormalize_latents(predicted_sv_lq)
                        # predicted_sv_uq_unnorm = self.mean_unnormalize_latents(predicted_sv_uq)
                        fake_y_hat, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                            latent_codes=torch.randn_like(true_w[:, 0, :]), style_vectors=predicted_sv,
                            input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                        )
                        fake_y_hat_lq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                            latent_codes=torch.randn_like(true_w[:, 0, :]), style_vectors=self.splice_latent_code(predicted_sv, predicted_sv_lq),
                            input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                        )
                        fake_y_hat_uq, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                            latent_codes=torch.randn_like(true_w[:, 0, :]), style_vectors=self.splice_latent_code(predicted_sv, predicted_sv_uq),
                            input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                        )
                        fake_y, _ = self.pretrained_psp_net.decoder.forward_with_style_vectors(
                            latent_codes=torch.randn_like(true_w[:, 0, :]), style_vectors=self.stylespace_flat_to_list(true_sv.squeeze()),
                            input_is_latent=True, randomize_noise=False, return_latents=True, modulate=False
                        )
                        fake_y_hat = self.face_pool(fake_y_hat)
                        fake_y_hat_lq = self.face_pool(fake_y_hat_lq)
                        fake_y_hat_uq = self.face_pool(fake_y_hat_uq)
                        fake_y = self.face_pool(fake_y)

                    self.parse_and_log_images(x=x, y=fake_y, y_hat=fake_y_hat, y_hat_lq=fake_y_hat_lq,
                                              y_hat_uq=fake_y_hat_uq, title='images/train/faces', display_count=2)

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0:
                    self.wb_logger.log_quantile_outputs_to_wandb(
                        x=x, y=fake_y, y_hat=fake_y_hat, y_hat_lq=fake_y_hat_lq, y_hat_uq=fake_y_hat_uq, id_logs=None,
                        prefix="train", step=self.global_step, opts=self.opts
                    )

                # Validation related
                val_loss_dict = None
                # if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                #     val_loss_dict = self.validate()
                #     if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                #         self.best_val_loss = val_loss_dict['loss']
                #         self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
