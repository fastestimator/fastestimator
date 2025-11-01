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
from typing import TypeVar

import torch

from fastestimator.backend._reduce_mean import reduce_mean

Tensor = TypeVar('Tensor', bound=torch.Tensor)


def l1_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate Mean Absolute Error between two tensors.

    This method can be used with PyTorch tensors:
    ```python

    true = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    L1 = fe.backend.l1_loss(y_pred=pred, y_true=true)                                         #[0.0750, 0.1500, 0.1000, 0.0000]

    true = torch.tensor([[1], [3], [2], [0]])
    pred = torch.tensor([[2.0], [0.0], [2.0], [1.0]])
    L1 = fe.backend.l1_loss(y_pred=pred, y_true=true)                                         #[1., 3., 0., 1.]
    ```

    Args:
        y_true: Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype: int, float16, float32.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.

    Returns:
        The L1 loss between `y_true` and `y_pred`.

    Raises:
        ValueError: If `y_pred` is an unacceptable data type.
    """
    if isinstance(y_pred, torch.Tensor):
        mae = reduce_mean(torch.nn.L1Loss(reduction="none")(y_pred, y_true), axis=-1)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(y_pred)))
    return mae
