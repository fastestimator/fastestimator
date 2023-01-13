# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
"""U-Net3d 3plus example."""
import tempfile
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_ as he_normal

import fastestimator as fe
from fastestimator.dataset.data.em_3d import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, VerticalFlip
from fastestimator.op.numpyop.univariate import ChannelTranspose, Minmax
from fastestimator.op.numpyop.univariate.expand_dims import ExpandDims
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.resize3d import Resize3D
from fastestimator.trace.adapt import EarlyStopping, ReduceLROnPlateau
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice


class StdSingleConvBlock(nn.Module):
    """A UNet3D StdSingleConvBlock block.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.BatchNorm3d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same"))

        for layer in self.layers:
            if isinstance(layer, nn.Conv3d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out


class ConvBlock(nn.Module):
    """A UNet3D ConvBlock block.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same"))

        for layer in self.layers:
            if isinstance(layer, nn.Conv3d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out


class StdDoubleConvBlock(nn.Module):
    """A UNet3D StdDoubleConvBlock block.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            StdSingleConvBlock(in_channels, out_channels),
            StdSingleConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out


class StdConvBlockSkip(nn.Module):
    """A UNet3D StdConvBlockSkip block skipping batch normalization.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(in_channels, out_channels),
                                    StdSingleConvBlock(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out


class UpsampleBlock(nn.Module):
    """A UNet3D UpsampleBlock block.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
        scale_factor: scale factor to up sample
        kernel_size: size of the kernel
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size, padding="same"),
        )

        for layer in self.layers:
            if isinstance(layer, nn.Conv3d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out


class DownSampleBlock(nn.Module):
    """A UNet3D DownSampleBlock block.

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
        scale_factor: scale factor to down sample
        kernel_size: size of the kernel
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.layers = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding="same"))

        for layer in self.layers:
            if isinstance(layer, nn.Conv3d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(F.max_pool3d(x, self.scale_factor))
        return out


class UNet3D3Plus(nn.Module):
    """A Attention UNet3D 3plus implementation in PyTorch.

    Args:
        input_size: The size of the input tensor (channels, height, width).
        output_channel: The number of output channels.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[1] or `input_size`[2] is not a multiple of 16.
    """
    def __init__(self,
                 input_size: Tuple[int, int, int, int] = (1, 128, 128, 24),
                 output_channel: int = 1,
                 channels: int = 64) -> None:
        UNet3D3Plus._check_input_size(input_size)
        super().__init__()
        self.input_size = input_size
        self.enc1 = StdConvBlockSkip(in_channels=input_size[0], out_channels=channels)
        self.enc2 = StdDoubleConvBlock(in_channels=channels, out_channels=channels * 2)
        self.enc3 = StdDoubleConvBlock(in_channels=channels * 2, out_channels=channels * 4)
        self.bottle_neck = StdDoubleConvBlock(in_channels=channels * 4, out_channels=channels * 8)

        self.up5_4 = UpsampleBlock(in_channels=channels * 8, out_channels=channels, scale_factor=2)
        self.up5_3 = ConvBlock(in_channels=channels * 4, out_channels=channels)
        self.down5_2 = DownSampleBlock(in_channels=channels * 2, out_channels=channels, scale_factor=2)
        self.down5_3 = DownSampleBlock(in_channels=channels, out_channels=channels, scale_factor=4)

        self.conv5 = StdSingleConvBlock(in_channels=channels * 4, out_channels=4 * channels)

        self.up6_4 = UpsampleBlock(in_channels=channels * 8, out_channels=channels, scale_factor=4)
        self.up6_3 = UpsampleBlock(in_channels=channels * 4, out_channels=channels, scale_factor=2)
        self.up6_2 = ConvBlock(in_channels=channels * 2, out_channels=channels)
        self.down6_1 = DownSampleBlock(in_channels=channels, out_channels=channels, scale_factor=2)

        self.conv6 = StdSingleConvBlock(in_channels=channels * 4, out_channels=4 * channels)

        self.up7_4 = UpsampleBlock(in_channels=channels * 8, out_channels=channels, scale_factor=8)
        self.up7_3 = UpsampleBlock(in_channels=channels * 4, out_channels=channels, scale_factor=4)
        self.up7_2 = UpsampleBlock(in_channels=channels * 4, out_channels=channels, scale_factor=2)
        self.conv7_1 = ConvBlock(in_channels=channels, out_channels=channels)

        self.conv7 = StdSingleConvBlock(in_channels=channels * 4, out_channels=4 * channels)

        self.dec1 = nn.Sequential(nn.BatchNorm3d(channels * 4),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(channels * 4, output_channel, 1, padding="same"),
                                  nn.Sigmoid())

        for layer in self.dec1:
            if isinstance(layer, nn.Conv3d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.enc1(x)
        pool1 = F.max_pool3d(conv1, 2)

        conv2 = self.enc2(pool1)
        pool2 = F.max_pool3d(conv2, 2)

        conv3 = self.enc3(pool2)
        pool3 = F.max_pool3d(conv3, 2)

        conv4 = self.bottle_neck(pool3)

        up5_4 = self.up5_4(conv4)
        up5_3 = self.up5_3(conv3)
        down5_2 = self.down5_2(conv2)
        down5_3 = self.down5_3(conv1)

        conv5 = self.conv5(torch.cat((up5_4, up5_3, down5_2, down5_3), 1))

        up6_4 = self.up6_4(conv4)
        up6_3 = self.up6_3(conv5)
        up6_2 = self.up6_2(conv2)
        down6_1 = self.down6_1(conv1)

        conv6 = self.conv6(torch.cat((up6_4, up6_3, up6_2, down6_1), 1))

        up7_4 = self.up7_4(conv4)
        up7_3 = self.up7_3(conv5)
        up7_2 = self.up7_2(conv6)
        conv7_1 = self.conv7_1(conv1)

        x_out = self.dec1(self.conv7(torch.cat((up7_4, up7_3, up7_2, conv7_1), 1)))
        return x_out

    @staticmethod
    def _check_input_size(input_size):
        if len(input_size) != 4:
            raise ValueError("Length of `input_size` is not 4 (channel, height, width, depth)")

        _, height, width, depth = input_size

        if height < 8 or not (height / 8.0).is_integer() or width < 8 or not (
                width / 8.0).is_integer() or depth < 8 or not (depth / 8.0).is_integer():
            raise ValueError(
                "All three height, width and depth of input_size need to be multiples of 8 (8, 16, 32, 48...)")


def get_estimator(epochs=40,
                  batch_size=1,
                  input_shape=(256, 256, 24),
                  channels=1,
                  num_classes=7,
                  filters=64,
                  learning_rate=1e-3,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  log_steps=20,
                  data_dir=None):

    # step 1
    train_data, eval_data = load_data(data_dir)

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="label", mode='train')),
            Sometimes(numpy_op=VerticalFlip(image_in="image", mask_in="label", mode='train')),
            Minmax(inputs="image", outputs="image"),
            ExpandDims(inputs="image", outputs="image"),
            ChannelTranspose(inputs=("image", "label"), outputs=("image", "label"), axes=(3, 0, 1, 2))
        ])

    # step 2
    model = fe.build(model_fn=lambda: UNet3D3Plus((channels, ) + input_shape, num_classes, filters),
                     optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=learning_rate),
                     model_name="unet3d_3plus")

    network = fe.Network(ops=[
        Resize3D(inputs="image", outputs="image", output_shape=input_shape),
        Resize3D(inputs="label", outputs="label", output_shape=input_shape, mode='!infer'),
        ModelOp(inputs="image", model=model, outputs="pred_segment"),
        CrossEntropy(inputs=("pred_segment", "label"), outputs="ce_loss", form="binary"),
        UpdateOp(model=model, loss_name="ce_loss")
    ])

    # step 3
    traces = [
        Dice(
            true_key="label",
            pred_key="pred_segment",
            channel_mapping={
                0: 'Cell',
                1: 'Mitochondria',
                2: 'AlphaGranule',
                3: 'CanalicularVessel',
                4: 'GranuleBody',
                5: 'GranuleCore'
            }),
        ReduceLROnPlateau(model=model, metric="Dice", patience=4, factor=0.5, best_mode="max"),
        BestModelSaver(model=model, save_dir=save_dir, metric='Dice', save_best_mode='max'),
        EarlyStopping(monitor="Dice", compare='max', min_delta=0.005, patience=6),
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             log_steps=log_steps,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
