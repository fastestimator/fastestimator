# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data.horse2zebra import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, RandomCrop, Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import ModelSaver


def lr_schedule(epoch):
    """Learning rate schedule"""
    if epoch <= 100:
        lr = 2e-4
    else:
        lr = 2e-4 * (200 - epoch) / 100
    return lr


class Buffer(TensorOp):
    """Image Buffer implementation as outlined in https://arxiv.org/abs/1703.10593v6"""
    def __init__(self, image_in=None, image_out=None, mode=None, buffer_size=50):
        super().__init__(inputs=image_in, outputs=image_out, mode=mode)
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.image_buffer = []

    def forward(self, data, state):
        output = []
        for image in data:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.image_buffer.append(image)
                output.append(image)
                self.num_imgs += 1
            else:
                if np.random.uniform() > 0.5:
                    idx = np.random.randint(self.buffer_size)
                    temp = self.image_buffer[idx].clone()
                    self.image_buffer[idx] = image
                    output.append(temp)
                else:
                    output.append(image)

        output = torch.cat(output, 0)
        return output


class GLoss(TensorOp):
    """TensorOp to compute generator loss"""
    def __init__(self, inputs, weight, device, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.LAMBDA = weight
        self.device = device
        self.average_loss = average_loss

    def _adversarial_loss(self, fake_img):
        return torch.mean(self.loss_fn(fake_img, torch.ones_like(fake_img, device=self.device)), dim=(2, 3))

    def _identity_loss(self, real_img, same_img):
        return 0.5 * self.LAMBDA * torch.mean(torch.abs(real_img - same_img), dim=(1, 2, 3))

    def _cycle_loss(self, real_img, cycled_img):
        return self.LAMBDA * torch.mean(torch.abs(real_img - cycled_img), dim=(1, 2, 3))

    def forward(self, data, state):
        real_img, fake_img, cycled_img, same_img = data
        total_loss = self._adversarial_loss(fake_img) + self._identity_loss(real_img, same_img) + self._cycle_loss(
            real_img, cycled_img)

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return total_loss


class DLoss(TensorOp):
    """TensorOp to compute discriminator loss"""
    def __init__(self, inputs, device, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.device = device
        self.average_loss = average_loss

    def forward(self, data, state):
        real_img, fake_img = data
        real_img_loss = torch.mean(self.loss_fn(real_img, torch.ones_like(real_img, device=self.device)), dim=(2, 3))
        fake_img_loss = torch.mean(self.loss_fn(fake_img, torch.zeros_like(real_img, device=self.device)), dim=(2, 3))
        total_loss = real_img_loss + fake_img_loss

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return 0.5 * total_loss


class ResidualBlock(nn.Module):
    """Residual block architecture"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
                                    nn.InstanceNorm2d(out_channels))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x_out = self.layers(x)
        x_out = x_out + x
        return x_out


class Discriminator(nn.Module):
    """Discriminator network architecture"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(256, 512, kernel_size=4, stride=1),
                                    nn.InstanceNorm2d(512),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 1, kernel_size=4, stride=1))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.layers(x)
        return x


class Generator(nn.Module):
    """Generator network architecture"""
    def __init__(self, num_blocks=9):
        super().__init__()
        self.layers1 = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, 64, kernel_size=7, stride=1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                     nn.InstanceNorm2d(256),
                                     nn.ReLU(inplace=True))
        self.resblocks = nn.Sequential(*[ResidualBlock(256, 256) for i in range(num_blocks)])
        self.layers2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.InstanceNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(3),
                                     nn.Conv2d(64, 3, kernel_size=7, stride=1))

        for block in [self.layers1, self.layers2]:
            for layer in block:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                    normal_(layer.weight.data, mean=0, std=0.02)

    def forward(self, x):
        x = self.layers1(x)
        x = self.resblocks(x)
        x = self.layers2(x)
        x = torch.tanh(x)
        return x


def get_estimator(weight=10.0,
                  epochs=200,
                  batch_size=1,
                  max_train_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir=None):
    train_data, _ = load_data(batch_size=batch_size, root_dir=data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = fe.Pipeline(
        train_data=train_data,
        ops=[
            ReadImage(inputs=["A", "B"], outputs=["A", "B"]),
            Normalize(inputs=["A", "B"], outputs=["real_A", "real_B"], mean=1.0, std=1.0, max_pixel_value=127.5),
            Resize(height=286, width=286, image_in="real_A", image_out="real_A", mode="train"),
            RandomCrop(height=256, width=256, image_in="real_A", image_out="real_A", mode="train"),
            Resize(height=286, width=286, image_in="real_B", image_out="real_B", mode="train"),
            RandomCrop(height=256, width=256, image_in="real_B", image_out="real_B", mode="train"),
            Sometimes(HorizontalFlip(image_in="real_A", image_out="real_A", mode="train")),
            Sometimes(HorizontalFlip(image_in="real_B", image_out="real_B", mode="train")),
            ChannelTranspose(inputs=["real_A", "real_B"], outputs=["real_A", "real_B"])
        ])

    g_AtoB = fe.build(model_fn=Generator,
                      model_name="g_AtoB",
                      optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
    g_BtoA = fe.build(model_fn=Generator,
                      model_name="g_BtoA",
                      optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
    d_A = fe.build(model_fn=Discriminator,
                   model_name="d_A",
                   optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))
    d_B = fe.build(model_fn=Discriminator,
                   model_name="d_B",
                   optimizer_fn=lambda x: torch.optim.Adam(x, lr=2e-4, betas=(0.5, 0.999)))

    network = fe.Network(ops=[
        ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
        ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
        Buffer(image_in="fake_A", image_out="buffer_fake_A"),
        Buffer(image_in="fake_B", image_out="buffer_fake_B"),
        ModelOp(inputs="real_A", model=d_A, outputs="d_real_A"),
        ModelOp(inputs="fake_A", model=d_A, outputs="d_fake_A"),
        ModelOp(inputs="buffer_fake_A", model=d_A, outputs="buffer_d_fake_A"),
        ModelOp(inputs="real_B", model=d_B, outputs="d_real_B"),
        ModelOp(inputs="fake_B", model=d_B, outputs="d_fake_B"),
        ModelOp(inputs="buffer_fake_B", model=d_B, outputs="buffer_d_fake_B"),
        ModelOp(inputs="real_A", model=g_BtoA, outputs="same_A"),
        ModelOp(inputs="fake_B", model=g_BtoA, outputs="cycled_A"),
        ModelOp(inputs="real_B", model=g_AtoB, outputs="same_B"),
        ModelOp(inputs="fake_A", model=g_AtoB, outputs="cycled_B"),
        GLoss(inputs=("real_A", "d_fake_B", "cycled_A", "same_A"), weight=weight, device=device, outputs="g_AtoB_loss"),
        GLoss(inputs=("real_B", "d_fake_A", "cycled_B", "same_B"), weight=weight, device=device, outputs="g_BtoA_loss"),
        DLoss(inputs=("d_real_A", "buffer_d_fake_A"), outputs="d_A_loss", device=device),
        DLoss(inputs=("d_real_B", "buffer_d_fake_B"), outputs="d_B_loss", device=device),
        UpdateOp(model=g_AtoB, loss_name="g_AtoB_loss"),
        UpdateOp(model=g_BtoA, loss_name="g_BtoA_loss"),
        UpdateOp(model=d_A, loss_name="d_A_loss"),
        UpdateOp(model=d_B, loss_name="d_B_loss")
    ])

    traces = [
        ModelSaver(model=g_AtoB, save_dir=save_dir, frequency=10),
        ModelSaver(model=g_BtoA, save_dir=save_dir, frequency=10),
        LRScheduler(model=g_AtoB, lr_fn=lr_schedule),
        LRScheduler(model=g_BtoA, lr_fn=lr_schedule),
        LRScheduler(model=d_A, lr_fn=lr_schedule),
        LRScheduler(model=d_B, lr_fn=lr_schedule)
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
