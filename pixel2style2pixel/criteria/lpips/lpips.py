import torch
import torch.nn as nn
import torch.nn.functional as F


from criteria.lpips.networks import get_network, LinLayers
from criteria.lpips.utils import get_state_dict

import sys
sys.path.insert(0, '/home/gridsan/swamiviv/projects/stylegan2-ada-pytorch')
import dnnlib


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network('alex').to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction='mean'):

        feat_x, feat_y = self.net(x), self.net(y)
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        if reduction == 'mean':
            return torch.sum(torch.cat(res, 0)) / x.shape[0]
        else:
            return torch.sum(torch.cat(res, 0))


class LPIPS_sg2(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, device):
        super(LPIPS_sg2, self).__init__()
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)
        self.resize_to_256 = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = (x + 1) * (255 / 2)
        y = (y + 1) * (255 / 2)
        x_features = self.vgg16(self.resize_to_256(x), resize_images=False, return_lpips=True)
        y_features = self.vgg16(self.resize_to_256(y), resize_images=False, return_lpips=True)

        return (x_features - y_features).square().sum() / x.shape[0]