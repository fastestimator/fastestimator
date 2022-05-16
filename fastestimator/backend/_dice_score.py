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

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._reduce_sum import reduce_sum

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)

allowed_data_types = [
    torch.float,
    torch.float16,
    torch.float32,
    torch.float64,
    np.float,
    np.float16,
    np.float32,
    np.float64,
    tf.float16,
    tf.float32,
    tf.float64
]


def get_denominator(y_true: Tensor, y_pred: Tensor, soft_dice: bool) -> Tensor:
    """
        Calculate sum/squared sum of y_true and y_pred

        Args:
            y_pred: Prediction with a shape like (Batch, C, H, W) for torch and (Batch, H, W, C) for tensorflow or numpy. dtype: float32 or float16.
            y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
            soft_dice: Whether to add direct sum or square sum of inputs

        Return:
            The sum or squared sum of y_pred and y_true
    """
    if soft_dice:
        return y_true**2 + y_pred**2
    else:
        return y_true + y_pred


def get_axis(y_true: Tensor, channel_average: bool) -> Tensor:
    """
        Get the axis to apply reduced_sum on.

        Args:
            y_true: Ground truth class labels with a shape like (Batch, C, H, W) for torch and (Batch, H, W, C) for tensorflow or numpy. dtype: int or float32 or float16.
            channel_average: Whether to average the channel wise dice score.

        Returns:
            The axis on which reduce_sum needs to be applied.
    """
    dims = len(y_true.shape)
    if dims <= 2:
        return None
    else:
        input_axis = list(range(dims))
        axis = input_axis[1:]
        if channel_average:
            if tf.is_tensor(y_true) or isinstance(y_true, np.ndarray):
                axis = input_axis[1:-1]
            elif isinstance(y_true, torch.Tensor):
                axis = input_axis[2:]
            else:
                raise ValueError("Unsupported tensor type.")

        return axis


def cast(y_true, epsilon, dtype):
    """
        Cast y_true, epsilon to desired data type.

        Args:
            y_true: Ground truth class labels with a shape like (Batch, C, H, W) for torch and (Batch, H, W, C) for tensorflow or numpy. dtype: int or float32 or float16.
            epsilon: Floating point value to avoid divide by zero error.
            dtype: Datatype to which the y_true and epsilon should be converted to.

        Returns:
            Converted y_true and epsilon values.

        Raises:
            AssertionError: If `y_true` are unacceptable data types. if data type is other than np.array, tensor.Tensor, tf.Tensor.
    """
    if dtype not in allowed_data_types:
        raise ValueError("Provided datatype {} is not supported, only {} data types are supported".format(
            dtype, allowed_data_types))

    if tf.is_tensor(y_true):
        return tf.cast(y_true, dtype), tf.cast(epsilon, dtype)
    elif isinstance(y_true, torch.Tensor):
        return y_true.type(dtype), torch.tensor(epsilon).type(dtype)
    elif isinstance(y_true, np.ndarray):
        return np.array(y_true, dtype=dtype), np.array(epsilon, dtype=dtype)
    else:
        raise ValueError("Unsupported tensor type.")


def dice_score(y_pred: Tensor,
               y_true: Tensor,
               soft_dice: bool = False,
               sample_average: bool = False,
               channel_average: bool = False,
               epsilon: float = 1e-6) -> Tensor:
    """
    Compute Dice score.

    This method can be used with Numpy data:
    ```python
    true = np.array([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = np.array([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # 0.161

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = tf.constant([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # 0.161
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = torch.tensor([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # 0.161
    ```

    ```
    Args:
        y_pred: Prediction with a shape like (Batch, C, H, W) for torch and (Batch, H, W, C) for tensorflow or numpy. dtype: float32 or float16.
        y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
        soft_dice: Whether to square elements. If True, square of elements is added.
        sample_average: Whether to average the element-wise dice score.
        channel_average: Whether to average the channel wise dice score.
        epsilon: floating point value to avoid divide by zero error.

    Returns:
        The dice score between `y_pred` and `y_true`. A scalar if `average_sample` is True, else a tensor with the shape (Batch).

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types. if data type is other than np.array, tensor.Tensor, tf.Tensor.
    """
    y_true, epsilon = cast(y_true, epsilon, y_pred.dtype)

    axis = get_axis(y_true, channel_average)

    keep_dims = False
    if axis == None:
        keep_dims = True

    numerator = reduce_sum(y_true * y_pred, axis=axis, keepdims=keep_dims)

    denominator = get_denominator(y_true, y_pred, soft_dice)

    denominator = reduce_sum(denominator, axis=axis, keepdims=keep_dims)

    dice_score = (2 * numerator) / (denominator + epsilon)

    if channel_average:
        dice_score = reduce_mean(dice_score, axis=-1)

    if sample_average:
        dice_score = reduce_mean(dice_score)

    return dice_score
