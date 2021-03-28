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
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn


class LeNet(torch.nn.Module):
    """A standard LeNet implementation in pytorch.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    The LeNet model has 3 convolution layers and 2 dense layers.

    Args:
        input_shape: The shape of the model input (channels, height, width).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_shape` is not 3.
        ValueError: `input_shape`[1] or `input_shape`[2] is smaller than 18.
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 28, 28), classes: int = 10) -> None:
        LeNet._check_input_shape(input_shape)
        super().__init__()
        conv_kernel = 3
        self.pool_kernel = 2
        self.conv1 = nn.Conv2d(input_shape[0], 32, conv_kernel)
        self.conv2 = nn.Conv2d(32, 64, conv_kernel)
        self.conv3 = nn.Conv2d(64, 64, conv_kernel)
        flat_x = ((((input_shape[1] - (conv_kernel - 1)) // self.pool_kernel) -
                   (conv_kernel - 1)) // self.pool_kernel) - (conv_kernel - 1)
        flat_y = ((((input_shape[2] - (conv_kernel - 1)) // self.pool_kernel) -
                   (conv_kernel - 1)) // self.pool_kernel) - (conv_kernel - 1)
        self.fc1 = nn.Linear(flat_x * flat_y * 64, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fn.relu(self.conv1(x))
        x = fn.max_pool2d(x, self.pool_kernel)
        x = fn.relu(self.conv2(x))
        x = fn.max_pool2d(x, self.pool_kernel)
        x = fn.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = fn.relu(self.fc1(x))
        x = fn.softmax(self.fc2(x), dim=-1)
        return x

    @staticmethod
    def _check_input_shape(input_shape):
        if len(input_shape) != 3:
            raise ValueError("Length of `input_shape` is not 3 (channel, height, width)")

        _, height, width = input_shape

        if height < 18 or width < 18:
            raise ValueError("Both height and width of input_shape need to not smaller than 18")
