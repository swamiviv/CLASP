
import matplotlib.pyplot as plt

import seaborn as sns
# sns.set_theme()
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.set_context("paper", rc={"font.size":12,"axes.titlesize":18,"axes.labelsize": 12, "font.weight":"bold"})
# sns.set(font_scale = 2)
# sns.set_style(style='white')

import bounds
import torch
import numpy as np
from copy import deepcopy

from matplotlib import rc, rcParams
# activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')

#rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

device='cuda' if torch.cuda.is_available() else 'cpu'


def splice_latent_code(latent_code, splice_dims, target_vector):
    spliced_latent_code = deepcopy(latent_code)
    if spliced_latent_code.ndim == 1:
        spliced_latent_code = spliced_latent_code.unsqueeze(0)
    spliced_latent_code[:, splice_dims] = target_vector
    return spliced_latent_code.squeeze()

def decorate_plot(ax=None, xlabel=None, ylabel=None, title_str=None, xticks=None):
    if ax is None:
        ax = plt.gca()
    small_fontsize = 12

    if xticks is not None:
        ax.xaxis.set_ticks(xticks)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(small_fontsize)
        #tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(small_fontsize)
        #tick.label1.set_fontweight('bold')

    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    if title_str:
        plt.title(title_str, fontsize=14)

def get_rcps_stats(val_idx_list, test_idx_list, indices_per_difficulty_level, lambdas, total_runs, all_losses, predicted_lower_edges,
                   predicted_upper_edges, alpha=0.1, delta=0.1):
    all_set_sizes = []
    mean_emp_risks = []
    all_emp_risks = []
    idx_lambda_calib_all = []
    for nrun in range(total_runs):
        val_idx = []
        test_idx = []
        for grade, indices_per_grade in indices_per_difficulty_level.items():
            for idx in val_idx_list[nrun]:
                val_idx.append(indices_per_grade[idx])
            for idx in test_idx_list[nrun]:
                test_idx.append(indices_per_grade[idx])
        val_loss_value_per_lambda = all_losses[:, val_idx]
        test_loss_value_per_lambda = all_losses[:, test_idx]

        # Compute RCPS bounds
        rhats = []
        rhats_plus = []
        lambda_calib = 0
        idx_lambda_calib_current = 0
        for idx, lam in enumerate(lambdas):
            losses = val_loss_value_per_lambda[idx]
            Rhat = losses.mean()
            RhatPlus = bounds.WSR_mu_plus(losses, delta)
            if RhatPlus <= alpha and Rhat <= alpha:
                lambda_calib = lam
                idx_lambda_calib_current = idx
                break
            rhats.append(Rhat)
            rhats_plus.append(RhatPlus)

        lower_edges = predicted_lower_edges[idx_lambda_calib_current]
        upper_edges = predicted_upper_edges[idx_lambda_calib_current]
        set_size_map = (upper_edges - lower_edges)

        emp_risk = test_loss_value_per_lambda[idx_lambda_calib_current]
        all_emp_risks.append(all_losses[idx_lambda_calib_current])
        mean_emp_risks.append(emp_risk.mean())
        all_set_sizes.append(set_size_map)
        idx_lambda_calib_all.append(idx_lambda_calib_current)

        # find lambda =1 index
        uncal_lambda_index = np.where(lambdas == 1)
        uncal_risk = test_loss_value_per_lambda[uncal_lambda_index].mean()

        print(f"\rRun {nrun} | Lambda: {lambda_calib:.4f}  |  AER: {emp_risk.mean():.4f}  |  RhatPlus: {RhatPlus:.4f} | Uncalibrated risk: {uncal_risk :.4f} | Calibrated risk: {emp_risk.mean() :.4f}")

    return all_set_sizes, all_emp_risks, mean_emp_risks, idx_lambda_calib_all

def interpolate(w0, w1, weights):
    if not torch.is_tensor(w0):
        w0 = torch.Tensor(w0)
        w1 = torch.Tensor(w1)
    w0, w1 = w0.to(device), w1.to(device)
    ws = []
    for weight in weights:
        ws.append(w0.lerp(w1, weight))
    return torch.vstack(ws).detach().cpu().numpy()