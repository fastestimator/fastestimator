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
import torch
import torch.nn as nn
import torch.nn.functional as fn

import numpy as np


class LeNet(torch.nn.Module):
    def __init__(self, n_channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = fn.softmax(x, dim=-1)
        return x
