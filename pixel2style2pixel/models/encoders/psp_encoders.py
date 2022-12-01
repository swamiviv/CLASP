import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear
import torchvision.models as models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, final_c = None):
        super(GradualStyleBlock, self).__init__()
        self.final_c = final_c
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)
        if self.final_c is not None:
            self.linear = EqualLinear(out_c, final_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class SimpleLinearBlock(Module):
    def __init__(self, in_c, out_c, final_c = None):
        super(SimpleLinearBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.final_c = final_c
        self.linear = EqualLinear(in_c, out_c, lr_mul=1, activation=True)
        if final_c is not None:
            self.linear_final = EqualLinear(out_c, final_c, lr_mul=1, activation=True)

    def forward(self, x):
        x = x.view(-1, self.in_c)
        x = self.linear(x)
        if self.final_c is not None:
            x = self.linear_final(x)
        return x

class ResidualGradualStyleBlock_1x1conv(Module):
    def __init__(self, in_c, out_c, spatial, final_c = None):
        super(ResidualGradualStyleBlock_1x1conv, self).__init__()
        self.final_c = final_c
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)
        if self.final_c is not None:
            self.linear = EqualLinear(out_c, final_c, lr_mul=1)

    def forward(self, x):
        input = x
        x = self.convs(x)
        x = input + x
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class GradualStyleBlock_1x1conv(Module):
    def __init__(self, in_c, out_c, spatial, final_c = None):
        super(GradualStyleBlock_1x1conv, self).__init__()
        self.final_c = final_c
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)
        if self.final_c is not None:
            self.linear = EqualLinear(out_c, final_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                    modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class GradualStyleEncoder_with_quantiles(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder_with_quantiles, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                    modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles_uq.append(style)
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles_lq.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        latents_lq = []
        latents_uq = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
            latents_lq.append(self.styles_lq[j](c3))
            latents_uq.append(self.styles_uq[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
            latents_lq.append(self.styles_lq[j](p2))
            latents_uq.append(self.styles_uq[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
            latents_lq.append(self.styles_lq[j](p1))
            latents_uq.append(self.styles_uq[j](p1))

        out_lq = torch.stack(latents_lq, dim=1)
        out_uq = torch.stack(latents_uq, dim=1)
        out_central = torch.stack(latents, dim=1)

        return torch.cat((out_lq.unsqueeze(1), out_central.unsqueeze(1), out_uq.unsqueeze(1)), axis=1)

class GradualStyleSpaceEncoder_with_quantiles(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleSpaceEncoder_with_quantiles, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                    modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.pooling_block = GradualStyleBlock(512, 512, 16)

        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_count = opts.n_style_space_vectors
        self.step_down_indices = [15, 18, 21, 24]
        self.coarse_ind = 9
        self.middle_ind = 18
        self.list_of_style_space_dims = [0]

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = SimpleLinearBlock(512, 512, final_c=512)
                self.list_of_style_space_dims.extend([512])
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = SimpleLinearBlock(512, 256, final_c=256)
                self.list_of_style_space_dims.extend([256])
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = SimpleLinearBlock(512, 128, final_c=128)
                self.list_of_style_space_dims.extend([128])
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = SimpleLinearBlock(512, 64, final_c=64)
                self.list_of_style_space_dims.extend([64])
            else:
                style = SimpleLinearBlock(512, 32, final_c=32)
                self.list_of_style_space_dims.extend([32])
            self.styles.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = SimpleLinearBlock(512, 512, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = SimpleLinearBlock(512, 256, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = SimpleLinearBlock(512, 128, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = SimpleLinearBlock(512, 64, final_c=64)
            else:
                style = SimpleLinearBlock(512, 32, final_c=32)
            self.styles_lq.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = SimpleLinearBlock(512, 512, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = SimpleLinearBlock(512, 256, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = SimpleLinearBlock(512, 128, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = SimpleLinearBlock(512, 64,  final_c=64)
            else:
                style = SimpleLinearBlock(512, 32, final_c=32)
            self.styles_uq.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        latents_lq = []
        latents_uq = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
        x = self.pooling_block(x)


        for j in range(self.coarse_ind):
            latents.append(self.styles[j](x))
            latents_lq.append(self.styles_lq[j](x))
            latents_uq.append(self.styles_uq[j](x))

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](x))
            latents_lq.append(self.styles_lq[j](x))
            latents_uq.append(self.styles_uq[j](x))

        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](x))
            latents_lq.append(self.styles_lq[j](x))
            latents_uq.append(self.styles_uq[j](x))

        return latents_lq, latents, latents_uq

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()
    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

class SimpleResnetEncoder_with_quantiles(Module):
    def __init__(self, num_layers, mode='ir', style_dims=9088, opts=None):
        super(SimpleResnetEncoder_with_quantiles, self).__init__()
        assert num_layers in [18, 50, 100, 152], 'num_layers should be 18, 50,100, or 152'
        self.num_layers = num_layers

        if num_layers == 18:
            self.body = models.resnet18(pretrained=False)
            self.body.fc = Identity()
        else:
            assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
            blocks = get_blocks(num_layers)
            if mode == 'ir':
                unit_module = bottleneck_IR
            elif mode == 'ir_se':
                unit_module = bottleneck_IR_SE
            self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                          BatchNorm2d(64),
                                          PReLU(64))
            modules = []
            for block in blocks:
                for bottleneck in block:
                        modules.append(unit_module(bottleneck.in_channel,
                                               bottleneck.depth,
                                               bottleneck.stride))
            self.body = Sequential(*modules)
        self.pooling_block = GradualStyleBlock(512, 512, 16)

        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_dims = style_dims

        self.projection = Sequential(
            Conv2d(512, style_dims, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            GlobalAveragePool2d(),
        )

        self.projection_lq = Sequential(
            Conv2d(512, style_dims, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            GlobalAveragePool2d(),
        )

        self.projection_uq = Sequential(
            Conv2d(512, style_dims, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            GlobalAveragePool2d(),
        )

    def forward(self, x):
        if self.num_layers == 18:
            x = self.body(x).unsqueeze(-1).unsqueeze(-1)
        else:
            x = self.input_layer(x)
            modulelist = list(self.body._modules.values())
            for i, l in enumerate(modulelist):
                x = l(x)

        x_pointwise = self.projection(x)
        x_lq = self.projection_lq(x)
        x_uq = self.projection_uq(x)

        return torch.cat((x_lq.unsqueeze(1), x_pointwise.unsqueeze(1), x_uq.unsqueeze(1)), axis=1)

class SimpleResnetEncoder_gaussian_output(Module):
    def __init__(self, num_layers, mode='ir', style_dims=9088, opts=None):
        super(SimpleResnetEncoder_gaussian_output, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.num_layers = num_layers
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                    modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.pooling_block = GradualStyleBlock(512, 512, 16)

        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_dims = style_dims

        self.mean_projection = Sequential(
            Conv2d(512, style_dims, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            GlobalAveragePool2d(),
        )

        self.var_projection = Sequential(
            Conv2d(512, style_dims, kernel_size=1),
            GlobalAveragePool2d(),
        )

    def forward(self, x):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
        x_mean = self.mean_projection(x)
        x_var = torch.relu(self.var_projection(x))

        return torch.cat((x_mean.unsqueeze(1), x_var.unsqueeze(1)), axis=1)

class ResidualFeaturetoStyleSpace_with_quantiles(Module):
    def __init__(self, opts=None):
        super(ResidualFeaturetoStyleSpace_with_quantiles, self).__init__()
        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_count = opts.n_style_space_vectors
        self.step_down_indices = [15, 18, 21, 24]
        self.coarse_ind = 9
        self.middle_ind = 18

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = ResidualGradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = ResidualGradualStyleBlock_1x1conv(256, 256, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = ResidualGradualStyleBlock_1x1conv(128, 128, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = ResidualGradualStyleBlock_1x1conv(64, 64, 4, final_c=64)
            else:
                style = ResidualGradualStyleBlock_1x1conv(32, 32, 4, final_c=32)
            self.styles.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = ResidualGradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = ResidualGradualStyleBlock_1x1conv(256, 256, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = ResidualGradualStyleBlock_1x1conv(128, 128, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = ResidualGradualStyleBlock_1x1conv(64, 64, 4, final_c=64)
            else:
                style = ResidualGradualStyleBlock_1x1conv(32, 32, 4, final_c=32)
            self.styles_lq.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = ResidualGradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = ResidualGradualStyleBlock_1x1conv(256, 256, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = ResidualGradualStyleBlock_1x1conv(128, 128, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = ResidualGradualStyleBlock_1x1conv(64, 64, 4, final_c=64)
            else:
                style = ResidualGradualStyleBlock_1x1conv(32, 32, 4, final_c=32)
            self.styles_uq.append(style)

    def forward(self, x):
        latents = []
        latents_lq = []
        latents_uq = []

        for j in range(len(x)):
            latents.append(self.styles[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))
            latents_lq.append(self.styles_lq[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))
            latents_uq.append(self.styles_uq[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))

        return latents_lq, latents, latents_uq


class FeaturetoStyleSpace_with_quantiles(Module):
    def __init__(self, opts=None):
        super(FeaturetoStyleSpace_with_quantiles, self).__init__()
        self.styles = nn.ModuleList()
        self.styles_lq = nn.ModuleList()
        self.styles_uq = nn.ModuleList()
        self.style_count = opts.n_style_space_vectors
        self.step_down_indices = [15, 18, 21, 24]
        self.coarse_ind = 9
        self.middle_ind = 18

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = GradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = GradualStyleBlock_1x1conv(256, 512, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = GradualStyleBlock_1x1conv(128, 512, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = GradualStyleBlock_1x1conv(64, 512, 4, final_c=64)
            else:
                style = GradualStyleBlock_1x1conv(32, 512, 4, final_c=32)
            self.styles.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = GradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = GradualStyleBlock_1x1conv(256, 512, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = GradualStyleBlock_1x1conv(128, 512, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = GradualStyleBlock_1x1conv(64, 512, 4, final_c=64)
            else:
                style = GradualStyleBlock_1x1conv(32, 512, 4, final_c=32)
            self.styles_lq.append(style)

        for i in range(self.style_count):
            if i < self.step_down_indices[0]:
                style = GradualStyleBlock_1x1conv(512, 512, 4, final_c=512)
            elif self.step_down_indices[0] <= i < self.step_down_indices[1]:
                style = GradualStyleBlock_1x1conv(256, 512, 4, final_c=256)
            elif self.step_down_indices[1] <= i < self.step_down_indices[2]:
                style = GradualStyleBlock_1x1conv(128, 512, 4, final_c=128)
            elif self.step_down_indices[2] <= i < self.step_down_indices[3]:
                style = GradualStyleBlock_1x1conv(64, 512, 4, final_c=64)
            else:
                style = GradualStyleBlock_1x1conv(32, 512, 4, final_c=32)
            self.styles_uq.append(style)

    def forward(self, x):
        latents = []
        latents_lq = []
        latents_uq = []

        for j in range(len(x)):
            latents.append(self.styles[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))
            latents_lq.append(self.styles_lq[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))
            latents_uq.append(self.styles_uq[j](x[j].squeeze().unsqueeze(-1).unsqueeze(-1)))

        return latents_lq, latents, latents_uq


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x


class QuantileEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(QuantileEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using QuantileEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x_lq = self.linear(x)
        x_lq = x_lq.view(-1, self.n_styles, 512)
        x_uq = self.linear(x)
        x_uq = x_uq.view(-1, self.n_styles, 512)
        x_central = self.linear(x)
        x_central = x_central.view(-1, self.n_styles, 512)
        return torch.cat((x_lq.unsqueeze(1), x_central.unsqueeze(1), x_uq.unsqueeze(1)), axis=1)