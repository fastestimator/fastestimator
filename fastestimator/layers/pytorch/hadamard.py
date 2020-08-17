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
import math
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
        max_prob: The maximum probability that can be assigned to a class. For numeric stability this must be less than
            1.0. Intuitively it makes sense to keep this close to 1, but to get adversarial training benefits it should
            be noticeably less than 1, for example 0.95 or even 0.8.
        power: The power parameter to be used by Inverse Distance Weighting when transforming Hadamard class distances
            into a class probability distribution. A value of 1.0 gives an intuitive mapping to probabilities, but small
            values such as 0.25 appear to give slightly better adversarial benefits. Large values like 2 or 3 give
            slightly faster convergence at the expense of adversarial performance. Must be greater than zero.

    Raises:
        ValueError: If `code_length` is invalid.
    """
    heads: Union[nn.ModuleList, nn.Module]

    def __init__(self,
                 in_features: Union[int, List[int]],
                 n_classes: int,
                 code_length: Optional[int] = None,
                 max_prob: float = 0.95,
                 power: float = 1.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        if code_length is None:
            code_length = max(16, 1 << (n_classes - 1).bit_length())
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}.")
        self.code_length = code_length
        if power <= 0:
            raise ValueError(f"power must be positive, but got {power}.")
        self.power = nn.Parameter(torch.tensor(power), requires_grad=False)
        if not 0.0 < max_prob < 1.0:
            raise ValueError(f"max_prob must be in the range (0, 1), but got {max_prob}")
        self.eps = nn.Parameter(
            torch.tensor(self.code_length * math.pow((1.0 - max_prob) / (max_prob * (self.n_classes - 1)), 1 / power)),
            requires_grad=False)
        labels = hadamard(self.code_length)
        # Cut off 0th column b/c it's constant. It would also be possible to make the column sign alternate, but that
        # would break the symmetry between rows in the code.
        labels = labels[:self.n_classes, 1:]
        self.labels = nn.Parameter(torch.tensor(labels, dtype=torch.float32), requires_grad=False)
        in_features = to_list(in_features)
        if len(in_features) > code_length - 1:
            raise ValueError(f"Too many input heads {len(in_features)} for the given code length {self.code_length}.")
        head_sizes = [self.code_length // len(in_features) for _ in range(len(in_features))]
        head_sizes[0] = head_sizes[0] + self.code_length - sum(head_sizes)
        head_sizes[0] = head_sizes[0] - 1  # We're going to cut off the 0th column from the code
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
        # Compute L1 distance
        x = torch.max(torch.sum(torch.abs(torch.unsqueeze(x, dim=1) - self.labels), dim=-1), self.eps)
        # Inverse Distance Weighting
        x = 1.0 / torch.pow(x, self.power)
        x = torch.div(x, torch.sum(x, dim=-1).view(-1, 1))
        return x
