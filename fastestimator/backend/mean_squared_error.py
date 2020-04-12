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
from typing import TypeVar

import tensorflow as tf
import torch

from fastestimator.backend.reduce_mean import reduce_mean

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate mean squared error between two tensors.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
    true = tf.constant([[1], [3], [2], [0]])
    pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
    true = tf.constant([[1], [3], [2], [0]])
    pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
    ```

    Args:
        y_true: Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype: int or float32.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32.

    Returns:
        The MSE between `y_true` and `y_pred`

    Raises:
        AssertionError: If `y_true` and `y_pred` have mismatched shapes or disparate types.
        ValueError: If `y_pred` is an unacceptable data type.
    """
    assert type(y_pred) == type(y_true), "y_pred and y_true must be of the same tensor type"
    assert y_pred.shape == y_true.shape, \
        f"MSE requires y_true and y_pred to have the same shape, but found {y_true.shape} and {y_pred.shape}"
    if isinstance(y_pred, tf.Tensor):
        mse = tf.losses.MSE(y_true, y_pred)
    elif isinstance(y_pred, torch.Tensor):
        mse = reduce_mean(
            torch.nn.MSELoss(reduction="none")(y_pred, y_true), axis=[ax for ax in range(y_pred.ndim)][1:])
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(y_pred)))
    return mse
