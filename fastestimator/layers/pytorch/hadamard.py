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
from typing import List, Optional, Union

import torch
import torch.nn as nn
from scipy.linalg import hadamard

from fastestimator.util.util import to_list


class HadamardCode(nn.Module):
    """A layer for applying an error correcting code to your outputs.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    See 'https://papers.nips.cc/paper/9070-error-correcting-output-codes-improve-probability-estimation-and-adversarial-
    robustness-of-deep-neural-networks'. Note that for best effectiveness, the model leading into this layer should be
    split into multiple independent chunks, whose outputs this layer can combine together in order to perform the code
    lookup.

    ```python
    # Use as a drop-in replacement for your softmax layer:
    def __init__(self, classes):
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, classes)
    def forward(self, x):
        x = fn.relu(self.fc1(x))
        x = fn.softmax(self.fc2(x), dim=-1)
    #   ----- vs ------
    def __init__(self, classes):
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = HadamardCode(64, classes)
    def forward(self, x):
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
    ```

    ```python
    # Use to combine multiple feature heads for a final output (biggest adversarial hardening benefit):
    def __init__(self, classes):
        self.fc1 = nn.ModuleList([nn.Linear(1024, 16) for _ in range(4)])
        self.fc2 = HadamardCode([16]*4, classes)
    def forward(self, x):
        x = [fn.relu(fc(x)) for fc in self.fc1]
        x = self.fc2(x)
    ```

    Args:
        in_features: How many input features there are (inputs should be of shape (Batch, N) or [(Batch, N), ...]).
        n_classes: How many output classes to map onto.
        code_length: How long of an error correcting code to use. Should be a positive multiple of 2. If not provided,
            the smallest power of 2 which is >= `n_outputs` will be used, or 16 if the latter is larger.

    Raises:
        ValueError: If `code_length` is invalid.
    """
    heads: Union[nn.ModuleList, nn.Module]

    def __init__(self, in_features: Union[int, List[int]], n_classes: int, code_length: Optional[int] = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        if code_length is None:
            code_length = max(16, 1 << (n_classes - 1).bit_length())
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}")
        self.code_length = code_length
        self.labels = nn.Parameter(
            torch.tensor(hadamard(self.code_length)[:self.n_classes], dtype=torch.float32).T, requires_grad=False)
        single_input = isinstance(in_features, int)
        in_features = to_list(in_features)
        if len(in_features) > code_length:
            raise ValueError(f"Too many input heads {len(in_features)} for the given code length {self.code_length}.")
        head_sizes = [self.code_length // len(in_features) for _ in range(len(in_features))]
        head_sizes[0] = head_sizes[0] + self.code_length - sum(head_sizes)
        self.heads = nn.ModuleList([
            nn.Linear(in_features=in_feat, out_features=out_feat) for in_feat, out_feat in zip(in_features, head_sizes)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # can't have forward function call subfunctions otherwise will fail on multi-gpu
        if isinstance(x, list):
            x = [head(tensor) for head, tensor in zip(self.heads, x)]
            x = torch.cat(x, dim=-1)
        else:
            x = self.heads[0](x)
        x = torch.tanh(x)
        x = torch.matmul(x, self.labels) + self.code_length
        x = torch.div(x, torch.sum(x, dim=1).view(-1, 1))
        return x
