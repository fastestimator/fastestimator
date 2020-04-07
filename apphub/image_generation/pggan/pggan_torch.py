import pdb

import numpy as np
import torch


def _nf(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class EqualizedLRDense(torch.nn.Linear):
    def __init__(self, in_features, out_features, gain=np.sqrt(2)):
        super().__init__(in_features, out_features, bias=False)
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0)
        self.wscale = np.float32(gain / np.sqrt(in_features))

    def forward(self, x):
        return super().forward(x) * self.wscale


class ApplyBias(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.bias = torch.nn.Parameter(torch.Tensor(in_features))
        torch.nn.init.constant_(self.bias.data, val=0.0)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x + self.bias.view(1, -1, 1, 1).expand_as(x)
        else:
            x = x + self.bias
        return x


class EqualizedLRConv2D(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros', gain=np.sqrt(2)):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0)
        fan_in = np.float32(np.prod(self.weight.data.shape[:-1]))
        self.wscale = np.float32(gain / np.sqrt(fan_in))

    def forward(self, x):
        return super().forward(x) * self.wscale


def pixel_normalization(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdims=True) + eps)


def mini_batch_std(x, group_size=4, eps=1e-8):
    b, c, h, w = x.shape
    group_size = min(group_size, b)
    y = x.reshape((group_size, -1, c, h, w))  # [G, M, C, H, W]
    y -= torch.mean(y, dim=0, keepdim=True)  # [G, M, C, H, W]
    y = torch.mean(y**2, axis=0)  # [M, C, H, W]
    y = torch.sqrt(y + eps)  # [M, C, H, W]
    y = torch.mean(y, dim=(1, 2, 3), keepdim=True)  # [M, 1, 1, 1]
    y = y.repeat(group_size, 1, h, w)  # [B, 1, H, W]
    return torch.cat((x, y), 1)


def fade_in(x, y, alpha):
    return (1 - alpha) * x + alpha * y


class ToRGB(torch.nn.Module):
    def __init__(self, in_channels, num_channels=3):
        super().__init__()
        self.elr_conv2d = EqualizedLRConv2D(in_channels, num_channels, kernel_size=1, padding=0, gain=1.0)
        self.bias = ApplyBias(in_features=num_channels)

    def forward(self, x):
        x = self.elr_conv2d(x)
        x = self.bias(x)
        return x


class FromRGB(torch.nn.Module):
    def __init__(self, res, num_channels=3):
        super().__init__()
        self.elr_conv2d = EqualizedLRConv2D(num_channels, _nf(res - 1), kernel_size=1, padding=0)
        self.bias = ApplyBias(in_features=_nf(res - 1))

    def forward(self, x):
        x = self.elr_conv2d(x)
        x = self.bias(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        return x


class BlockG1D(torch.nn.Module):
    def __init__(self, res=2, latent_dim=512):
        super().__init__()
        self.elr_dense = EqualizedLRDense(in_features=latent_dim, out_features=_nf(res - 1) * 16, gain=np.sqrt(2) / 4)
        self.bias1 = ApplyBias(in_features=_nf(res - 1))
        self.elr_conv2d = EqualizedLRConv2D(in_channels=_nf(res - 1), out_channels=_nf(res - 1))
        self.bias2 = ApplyBias(in_features=_nf(res - 1))
        self.res = res

    def forward(self, x):
        # x: [batch, 512]
        x = pixel_normalization(x)  # [batch, 512]
        x = self.elr_dense(x)  # [batch, _nf(res - 1) * 16]
        x = x.view(-1, _nf(self.res - 1), 4, 4)  # [batch, _nf(res - 1), 4, 4]
        x = self.bias1(x)  # [batch, _nf(res - 1), 4, 4]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, _nf(res - 1), 4, 4]
        x = pixel_normalization(x)  # [batch, _nf(res - 1), 4, 4]
        x = self.elr_conv2d(x)  # [batch, _nf(res - 1), 4, 4]
        x = self.bias2(x)  # [batch, _nf(res - 1), 4, 4]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, _nf(res - 1), 4, 4]
        x = pixel_normalization(x)
        return x


class BlockG2D(torch.nn.Module):
    def __init__(self, res):
        super().__init__()
        self.elr_conv2d1 = EqualizedLRConv2D(in_channels=_nf(res - 2), out_channels=_nf(res - 1))
        self.bias1 = ApplyBias(in_features=_nf(res - 1))
        self.elr_conv2d2 = EqualizedLRConv2D(in_channels=_nf(res - 1), out_channels=_nf(res - 1))
        self.bias2 = ApplyBias(in_features=_nf(res - 1))

    def forward(self, x):
        # x: [batch, _nf(res - 2), 2**(res - 1), 2**(res - 1)]
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2))  # [batch, _nf(res - 2), 2**res , 2**res)]
        x = self.elr_conv2d1(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = self.bias1(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = pixel_normalization(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = self.elr_conv2d2(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = self.bias2(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, _nf(res - 1), 2**res , 2**res)]
        x = pixel_normalization(x)  # [batch, _nf(res - 1), 2**res , 2**res)]
        return x


def _block_G(res, latent_dim=512, initial_resolution=2):
    if res == initial_resolution:
        model = BlockG1D(res=res, latent_dim=latent_dim)
    else:
        model = BlockG2D(res=res)
    return model


class Gen(torch.nn.Module):
    def __init__(self, g_blocks, rgb_blocks, fade_in_alpha):
        super().__init__()
        self.g_blocks = torch.nn.ModuleList(g_blocks)
        self.rgb_blocks = torch.nn.ModuleList(rgb_blocks)
        self.fade_in_alpha = fade_in_alpha

    def forward(self, x):
        for g in self.g_blocks[:-1]:
            x = g(x)
        previous_img = self.rgb_blocks[0](x)
        previous_img = torch.nn.functional.interpolate(previous_img, scale_factor=(2, 2))
        x = self.g_blocks[-1](x)
        new_img = self.rgb_blocks[1](x)
        return fade_in(previous_img, new_img, self.fade_in_alpha)


def build_G(fade_in_alpha, latent_dim=512, initial_resolution=2, target_resolution=10, num_channels=3):
    g_blocks = [
        _block_G(res, latent_dim, initial_resolution) for res in range(initial_resolution, target_resolution + 1)
    ]
    rgb_blocks = [ToRGB(_nf(res - 1), num_channels) for res in range(initial_resolution, target_resolution + 1)]
    generators = [torch.nn.Sequential(g_blocks[0], rgb_blocks[0])]
    for idx in range(2, len(g_blocks) + 1):
        generators.append(Gen(g_blocks[0:idx], rgb_blocks[idx - 2:idx], fade_in_alpha))
    final_model_list = g_blocks + [rgb_blocks[-1]]
    generators.append(torch.nn.Sequential(*final_model_list))
    return generators


class BlockD1D(torch.nn.Module):
    def __init__(self, res=2):
        super().__init__()
        self.elr_conv2d = EqualizedLRConv2D(in_channels=_nf(res - 1) + 1, out_channels=_nf(res - 1))
        self.bias1 = ApplyBias(in_features=_nf(res - 1))
        self.elr_dense1 = EqualizedLRDense(in_features=_nf(res - 1) * 16, out_features=_nf(res - 2))
        self.bias2 = ApplyBias(in_features=_nf(res - 2))
        self.elr_dense2 = EqualizedLRDense(in_features=_nf(res - 2), out_features=1, gain=1.0)
        self.bias3 = ApplyBias(in_features=1)
        self.res = res

    def forward(self, x):
        # x: [batch, 512, 4, 4]
        x = mini_batch_std(x)  # [batch, 513, 4, 4]
        x = self.elr_conv2d(x)  # [batch, 512, 4, 4]
        x = self.bias1(x)  # [batch, 512, 4, 4]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, 512, 4, 4]
        x = x.view(-1, _nf(self.res - 1) * 16)  # [batch, 512*4*4]
        x = self.elr_dense1(x)  # [batch, 512]
        x = self.bias2(x)  # [batch, 512]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)  # [batch, 512]
        x = self.elr_dense2(x)  # [batch, 1]
        x = self.bias3(x)  # [batch, 1]
        return x


class BlockD2D(torch.nn.Module):
    def __init__(self, res):
        super().__init__()
        self.elr_conv2d1 = EqualizedLRConv2D(in_channels=_nf(res - 1), out_channels=_nf(res - 1))
        self.bias1 = ApplyBias(in_features=_nf(res - 1))
        self.elr_conv2d2 = EqualizedLRConv2D(in_channels=_nf(res - 1), out_channels=_nf(res - 2))
        self.bias2 = ApplyBias(in_features=_nf(res - 2))
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.elr_conv2d1(x)
        x = self.bias1(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.elr_conv2d2(x)
        x = self.bias2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.pool(x)
        return x


def _block_D(res, initial_resolution=2):
    if res == initial_resolution:
        model = BlockD1D(res)
    else:
        model = BlockD2D(res)
    return model


class Disc(torch.nn.Module):
    def __init__(self, d_blocks, rgb_blocks, fade_in_alpha):
        super().__init__()
        self.d_blocks = torch.nn.ModuleList(d_blocks)
        self.rgb_blocks = torch.nn.ModuleList(rgb_blocks)
        self.fade_in_alpha = fade_in_alpha
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        new_x = self.rgb_blocks[1](x)
        new_x = self.d_blocks[-1](new_x)
        downscale_x = self.pool(x)
        downscale_x = self.rgb_blocks[0](downscale_x)
        x = fade_in(downscale_x, new_x, self.fade_in_alpha)
        for d in self.d_blocks[:-1][::-1]:
            x = d(x)
        return x


def build_D(fade_in_alpha, initial_resolution=2, target_resolution=10, num_channels=3):
    d_blocks = [_block_D(res, initial_resolution) for res in range(initial_resolution, target_resolution + 1)]
    rgb_blocks = [FromRGB(res, num_channels) for res in range(initial_resolution, target_resolution + 1)]
    discriminators = [torch.nn.Sequential(rgb_blocks[0], d_blocks[0])]
    for idx in range(2, len(d_blocks) + 1):
        discriminators.append(Disc(d_blocks[0:idx], rgb_blocks[idx - 2:idx], fade_in_alpha))
    return discriminators


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    pass
