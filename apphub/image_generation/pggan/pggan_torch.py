#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import torch
from fastestimator.backend import feed_forward, get_gradient
from fastestimator.dataset.data import nih_chestxray
from fastestimator.op import LambdaOp
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler
from fastestimator.trace import Trace
from fastestimator.trace.io import ModelSaver
from fastestimator.util import get_num_devices, traceable
from torch.optim import Adam


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
        fan_in = np.float32(np.prod(self.weight.data.shape[1:]))
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
    return (1.0 - alpha) * x + alpha * y


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
        self.upsample = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        # x: [batch, _nf(res - 2), 2**(res - 1), 2**(res - 1)]
        x = self.upsample(x)
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
        self.upsample = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        for g in self.g_blocks[:-1]:
            x = g(x)
        previous_img = self.rgb_blocks[0](x)
        previous_img = self.upsample(previous_img)
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


@traceable()
class ImageBlender(TensorOp):
    def __init__(self, alpha, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha

    def forward(self, data, state):
        image, image_lowres = data
        new_img = self.alpha * image + (1 - self.alpha) * image_lowres
        return new_img


@traceable()
class Interpolate(TensorOp):
    def forward(self, data, state):
        fake, real = data
        batch_size = real.shape[0]
        coeff = torch.rand(batch_size, 1, 1, 1).to(fake.device)
        return real + (fake - real) * coeff


@traceable()
class GradientPenalty(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        x_interp, interp_score = data
        gradient_x_interp = get_gradient(torch.sum(interp_score), x_interp, higher_order=True)
        grad_l2 = torch.sqrt(torch.sum(gradient_x_interp**2, dim=(1, 2, 3)))
        gp = (grad_l2 - 1.0)**2
        return gp


@traceable()
class GLoss(TensorOp):
    def forward(self, data, state):
        return -torch.mean(data)


@traceable()
class DLoss(TensorOp):
    """Compute discriminator loss."""
    def __init__(self, inputs, outputs=None, mode=None, wgan_lambda=10, wgan_epsilon=0.001):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.wgan_lambda = wgan_lambda
        self.wgan_epsilon = wgan_epsilon

    def forward(self, data, state):
        real_score, fake_score, gp = data
        loss = fake_score - real_score + self.wgan_lambda * gp + real_score**2 * self.wgan_epsilon
        return torch.mean(loss)


@traceable()
class AlphaController(Trace):
    def __init__(self, alpha, fade_start_epochs, duration, batch_scheduler, num_examples):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.alpha = alpha
        self.fade_start_epochs = fade_start_epochs
        self.duration = duration
        self.batch_scheduler = batch_scheduler
        self.num_examples = num_examples
        self.change_alpha = False
        self.nimg_total = self.duration * self.num_examples
        self._idx = 0
        self.nimg_so_far = 0
        self.current_batch_size = None

    def on_epoch_begin(self, state):
        # check whetehr the current epoch is in smooth transition of resolutions
        fade_epoch = self.fade_start_epochs[self._idx]
        if self.system.epoch_idx == fade_epoch:
            self.change_alpha = True
            self.nimg_so_far = 0
            self.current_batch_size = self.batch_scheduler.get_current_value(self.system.epoch_idx)
            print("FastEstimator-Alpha: Started fading in for size {}".format(2**(self._idx + 3)))
        elif self.system.epoch_idx == fade_epoch + self.duration:
            print("FastEstimator-Alpha: Finished fading in for size {}".format(2**(self._idx + 3)))
            self.change_alpha = False
            if self._idx + 1 < len(self.fade_start_epochs):
                self._idx += 1
            self.alpha.data = torch.tensor(1.0)

    def on_batch_begin(self, state):
        # if in resolution transition, smoothly change the alpha from 0 to 1
        if self.change_alpha:
            self.nimg_so_far += self.current_batch_size
            self.alpha.data = torch.tensor(self.nimg_so_far / self.nimg_total, dtype=torch.float32)


@traceable()
class ImageSaving(Trace):
    def __init__(self, epoch_model_map, save_dir, num_sample=16, latent_dim=512):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.epoch_model_map = epoch_model_map
        self.save_dir = save_dir
        self.latent_dim = latent_dim
        self.num_sample = num_sample
        self.eps = 1e-8

    def on_epoch_end(self, state):
        if self.system.epoch_idx in self.epoch_model_map:
            model = self.epoch_model_map[self.system.epoch_idx]
            for i in range(self.num_sample):
                random_vectors = torch.normal(
                    mean=0.0, std=1.0, size=(1, self.latent_dim)).to("cuda:0" if torch.cuda.is_available() else "cpu")
                pred = feed_forward(model, random_vectors, training=False)
                if torch.cuda.is_available():
                    pred = pred.to("cpu")
                disp_img = np.transpose(pred.data.numpy(), (0, 2, 3, 1))  # BCHW -> BHWC
                disp_img = np.squeeze(disp_img)
                disp_img -= disp_img.min()
                disp_img /= (disp_img.max() + self.eps)
                disp_img = np.uint8(disp_img * 255)
                cv2.imwrite(
                    os.path.join(self.save_dir, 'image_at_{:08d}_{}.png').format(self.system.epoch_idx, i), disp_img)
            print("on epoch {}, saving image to {}".format(self.system.epoch_idx, self.save_dir))


def get_estimator(target_size=128,
                  epochs=55,
                  save_dir=tempfile.mkdtemp(),
                  max_train_steps_per_epoch=None,
                  data_dir=None):
    # assert growth parameters
    num_grow = np.log2(target_size) - 2
    assert num_grow >= 1 and num_grow % 1 == 0, "need exponential of 2 and greater than 8 as target size"
    num_phases = int(2 * num_grow + 1)
    assert epochs % num_phases == 0, "epoch must be multiple of {} for size {}".format(num_phases, target_size)
    num_grow, phase_length = int(num_grow), int(epochs / num_phases)
    event_epoch = [1, 1 + phase_length] + [phase_length * (2 * i + 1) + 1 for i in range(1, num_grow)]
    event_size = [4] + [2**(i + 3) for i in range(num_grow)]
    # set up data schedules
    dataset = nih_chestxray.load_data(root_dir=data_dir)
    resize_map = {
        epoch: Resize(image_in="x", image_out="x", height=size, width=size)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    resize_low_res_map1 = {
        epoch: Resize(image_in="x", image_out="x_low_res", height=size // 2, width=size // 2)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    resize_low_res_map2 = {
        epoch: Resize(image_in="x_low_res", image_out="x_low_res", height=size, width=size)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    batch_size_map = {
        epoch: max(512 // size, 4) * get_num_devices() if size <= 512 else 2 * get_num_devices()
        for (epoch, size) in zip(event_epoch, event_size)
    }
    batch_scheduler = EpochScheduler(epoch_dict=batch_size_map)
    pipeline = fe.Pipeline(
        batch_size=batch_scheduler,
        train_data=dataset,
        drop_last=True,
        ops=[
            ReadImage(inputs="x", outputs="x", color_flag='gray'),
            EpochScheduler(epoch_dict=resize_map),
            EpochScheduler(epoch_dict=resize_low_res_map1),
            EpochScheduler(epoch_dict=resize_low_res_map2),
            Normalize(inputs=["x", "x_low_res"], outputs=["x", "x_low_res"], mean=1.0, std=1.0, max_pixel_value=127.5),
            ChannelTranspose(inputs=["x", "x_low_res"], outputs=["x", "x_low_res"]),
            LambdaOp(fn=lambda: np.random.normal(size=[512]).astype('float32'), outputs="z")
        ])
    fade_in_alpha = torch.tensor(1.0)
    d_models = fe.build(
        model_fn=lambda: build_D(fade_in_alpha, target_resolution=int(np.log2(target_size)), num_channels=1),
        optimizer_fn=[lambda x: Adam(x, lr=0.001, betas=(0.0, 0.99), eps=1e-8)] * len(event_size),
        model_name=["d_{}".format(size) for size in event_size])

    g_models = fe.build(
        model_fn=lambda: build_G(fade_in_alpha, target_resolution=int(np.log2(target_size)), num_channels=1),
        optimizer_fn=[lambda x: Adam(x, lr=0.001, betas=(0.0, 0.99), eps=1e-8)] * len(event_size) + [None],
        model_name=["g_{}".format(size) for size in event_size] + ["G"])
    fake_img_map = {
        epoch: ModelOp(inputs="z", outputs="x_fake", model=model)
        for (epoch, model) in zip(event_epoch, g_models[:-1])
    }
    fake_score_map = {
        epoch: ModelOp(inputs="x_fake", outputs="fake_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    real_score_map = {
        epoch: ModelOp(inputs="x_blend", outputs="real_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    interp_score_map = {
        epoch: ModelOp(inputs="x_interp", outputs="interp_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    g_update_map = {
        epoch: UpdateOp(loss_name="gloss", model=model)
        for (epoch, model) in zip(event_epoch, g_models[:-1])
    }
    d_update_map = {epoch: UpdateOp(loss_name="dloss", model=model) for (epoch, model) in zip(event_epoch, d_models)}
    network = fe.Network(ops=[
        EpochScheduler(fake_img_map),
        EpochScheduler(fake_score_map),
        ImageBlender(alpha=fade_in_alpha, inputs=("x", "x_low_res"), outputs="x_blend"),
        EpochScheduler(real_score_map),
        Interpolate(inputs=("x_fake", "x"), outputs="x_interp"),
        EpochScheduler(interp_score_map),
        GradientPenalty(inputs=("x_interp", "interp_score"), outputs="gp"),
        GLoss(inputs="fake_score", outputs="gloss"),
        DLoss(inputs=("real_score", "fake_score", "gp"), outputs="dloss"),
        EpochScheduler(g_update_map),
        EpochScheduler(d_update_map)
    ])
    traces = [
        AlphaController(alpha=fade_in_alpha,
                        fade_start_epochs=event_epoch[1:],
                        duration=phase_length,
                        batch_scheduler=batch_scheduler,
                        num_examples=len(dataset)),
        ModelSaver(model=g_models[-1], save_dir=save_dir, frequency=phase_length),
        ImageSaving(
            epoch_model_map={epoch - 1: model
                             for (epoch, model) in zip(event_epoch[1:] + [epochs + 1], g_models[:-1])},
            save_dir=save_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
