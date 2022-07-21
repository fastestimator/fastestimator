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

import tensorflow as tf
import torch

from fastestimator.backend._reduce_mean import reduce_mean

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def smooth_l1_loss(y_true: Tensor, y_pred: Tensor, beta: float = 1.0) -> Tensor:
    """Calculate Smooth L1 Loss between two tensors.

    This method can be used with TensorFlow tensors:
    ```python

    true = tf.constant([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    Smooth_L1 = fe.backend.smooth_l1_loss(y_pred=pred, y_true=true, loss_type='smooth', beta=0.65)   #[0.0048, 0.0269, 0.0125, 0.0000]

    true = tf.constant([[1], [3], [2], [0]])
    pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
    Smooth_L1 = fe.backend.smooth_l1_loss(y_pred=pred, y_true=true, loss_type='smooth', beta=0.65)   #[0.6750, 2.6750, 0.0000, 0.6750]
    ```

    This method can be used with PyTorch tensors:
    ```python

    true = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    Smooth_L1 = fe.backend.smooth_l1_loss(y_pred=pred, y_true=true, loss_type='smooth', beta=0.65)   #[0.0048, 0.0269, 0.0125, 0.0000]

    true = torch.tensor([[1], [3], [2], [0]])
    pred = torch.tensor([[2.0], [0.0], [2.0], [1.0]])
    Smooth_L1 = fe.backend.smooth_l1_loss(y_pred=pred, y_true=true, loss_type='smooth', beta=0.65)   #[0.6750, 2.6750, 0.0000, 0.6750]
    ```

    Args:
        y_true: Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype: int, float16, float32.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.
        beta: Threshold factor. Needs to be a positive number. dtype: float16 or float32.

    Returns:
        Smooth L1 between `y_true` and `y_pred` wrt beta.

    Raises:
        ValueError: If `y_pred` is an unacceptable data type.
        ValueError: If beta is less than 1 for Smooth L1 loss.
    """
    if beta <= 0:
        raise ValueError("Beta cannot be less than or equal to 0")

    if tf.is_tensor(y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        regression_diff = tf.math.abs(y_true - y_pred)  # |y - f(x)|
        regression_loss = tf.where(tf.math.less(regression_diff, beta),
                                   0.5 * tf.math.pow(regression_diff, 2) / beta,
                                   regression_diff - 0.5 * beta)
        smooth_mae = reduce_mean(regression_loss, axis=-1)
    elif isinstance(y_pred, torch.Tensor):
        smooth_mae = reduce_mean(
            torch.nn.SmoothL1Loss(reduction="none", beta=beta)(y_pred, y_true), axis=-1)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(y_pred)))
    return smooth_mae
