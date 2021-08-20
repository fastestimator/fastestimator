# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_ as he_normal


class UNetEncoderBlock(nn.Module):
    """A UNet encoder block.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    Args:
        in_channels: How many channels enter the encoder.
        out_channels: How many channels leave the encoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x)
        return out, F.max_pool2d(out, 2)


class UNetDecoderBlock(nn.Module):
    """A UNet decoder block.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    Args:
        in_channels: How many channels enter the decoder.
        mid_channels: How many channels are used for the decoder's intermediate layer.
        out_channels: How many channels leave the decoder.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(mid_channels, out_channels, 3, padding=1),
                                    nn.ReLU(inplace=True))

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionBlock(nn.Module):
    """An Attention block.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    Args:
        in_channels: How many channels enter the attention block.
        out_channels: How many channels leave the attention block.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dec_layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.enc_layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.attn_layers = nn.Sequential(nn.Conv2d(out_channels, 1, 1), nn.BatchNorm2d(1))

    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor) -> torch.Tensor:
        dec_out = self.dec_layers(decoder_input)
        enc_out = self.enc_layers(encoder_input)
        attn = F.relu(dec_out + enc_out)
        attn = self.attn_layers(attn)
        attn = torch.sigmoid(attn)
        return encoder_input * attn


class AttentionUNet(nn.Module):
    """Attention based UNet implementation in PyTorch.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    Args:
        input_size: The size of the input tensor (channels, height, width).

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[1] or `input_size`[2] is not a multiple of 16.
    """
    def __init__(self, input_size: Tuple[int, int, int] = (1, 128, 128)) -> None:
        AttentionUNet._check_input_size(input_size)
        super().__init__()
        self.input_size = input_size
        self.enc1 = UNetEncoderBlock(in_channels=input_size[0], out_channels=64)
        self.enc2 = UNetEncoderBlock(in_channels=64, out_channels=128)
        self.enc3 = UNetEncoderBlock(in_channels=128, out_channels=256)
        self.enc4 = UNetEncoderBlock(in_channels=256, out_channels=512)
        self.bottle_neck = UNetDecoderBlock(in_channels=512, mid_channels=1024, out_channels=512)
        self.attn4 = AttentionBlock(in_channels=512, out_channels=512)
        self.attn3 = AttentionBlock(in_channels=256, out_channels=256)
        self.attn2 = AttentionBlock(in_channels=128, out_channels=128)
        self.attn1 = AttentionBlock(in_channels=64, out_channels=64)
        self.dec4 = UNetDecoderBlock(in_channels=1024, mid_channels=512, out_channels=256)
        self.dec3 = UNetDecoderBlock(in_channels=512, mid_channels=256, out_channels=128)
        self.dec2 = UNetDecoderBlock(in_channels=256, mid_channels=128, out_channels=64)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1),
                                  nn.Sigmoid())

        for layer in self.dec1:
            if isinstance(layer, nn.Conv2d):
                he_normal(layer.weight.data)
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x_e1 = self.enc1(x)
        x2, x_e2 = self.enc2(x_e1)
        x3, x_e3 = self.enc3(x_e2)
        x4, x_e4 = self.enc4(x_e3)

        x_bottle_neck = self.bottle_neck(x_e4)
        x4_attn = self.attn4(x_bottle_neck, x4)
        x_d4 = self.dec4(torch.cat((x_bottle_neck, x4_attn), 1))
        x3_attn = self.attn3(x_d4, x3)
        x_d3 = self.dec3(torch.cat((x_d4, x3_attn), 1))
        x2_attn = self.attn2(x_d3, x2)
        x_d2 = self.dec2(torch.cat((x_d3, x2_attn), 1))
        x1_attn = self.attn1(x_d2, x1)
        x_out = self.dec1(torch.cat((x_d2, x1_attn), 1))
        return x_out

    @staticmethod
    def _check_input_size(input_size):
        if len(input_size) != 3:
            raise ValueError("Length of `input_size` is not 3 (channel, height, width)")

        _, height, width = input_size

        if height < 16 or not (height / 16.0).is_integer() or width < 16 or not (width / 16.0).is_integer():
            raise ValueError("Both height and width of input_size need to be multiples of 16 (16, 32, 48...)")
