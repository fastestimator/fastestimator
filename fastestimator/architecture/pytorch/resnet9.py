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
import torch.nn.functional as fn


class ResNet9(nn.Module):
    """A 9-layer ResNet PyTorch model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (channels, height, width). Both width and height of input_size should
            not be smaller than 16.
        classes: The number of outputs.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[1] or `input_size`[2] is not a multiple of 16.
    """
    def __init__(self, input_size: Tuple[int, int, int] = (3, 32, 32), classes: int = 10):
        ResNet9._check_input_size(input_size)
        super().__init__()
        self.conv0 = nn.Conv2d(input_size[0], 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.8)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.8)
        self.residual1 = Residual(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.8)
        self.residual2 = Residual(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.8)
        self.residual3 = Residual(512)
        self.fc1 = nn.Linear(512, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = fn.softmax(x, dim=-1)
        return x

    @staticmethod
    def _check_input_size(input_size):
        if len(input_size) != 3:
            raise ValueError("Length of `input_size` is not 3 (channel, height, width)")

        _, height, width = input_size

        if height < 16 or width < 16:
            raise ValueError("Both height and width of input_size need to not smaller than 16")


class Residual(nn.Module):
    """A two-layer unit for ResNet9. The output size is the same as input.

    Args:
        channel: Number of input channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x
