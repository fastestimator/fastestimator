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

from fastestimator.backend.convert_tensor_precision import convert_input_precision
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_sum import reduce_sum

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def get_denominator(y_true: Tensor, y_pred: Tensor, soft_dice: bool) -> Tensor:
    """
        Calculate sum/squared sum of y_true and y_pred

        Args:
            y_pred: Prediction with a shape like (Batch, C). dtype: float32 or float16.
            y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
            soft_dice: Whether to add direct sum or square sum of inputs

        Return:
            The sum or squared sum of y_pred and y_true
    """
    if soft_dice:
        if tf.is_tensor(y_true):
            return tf.square(y_true) + tf.square(y_pred)
        elif isinstance(y_true, torch.Tensor):
            return torch.square(y_true) + torch.square(y_pred)
        elif isinstance(y_true, np.ndarray):
            return np.square(y_true) + np.square(y_pred)
    else:
        return y_true + y_pred


def get_axis(y_true: Tensor, channel_average: bool) -> Tensor:
    """
        Get the axis to apply reduced_sum on.

        Args:
            y_true: Prediction with a shape like (Batch, C). dtype: float32 or float16.
            channel_average: Whether to average the channel wise dice score.

        Returns:
            The axis on which reduce_sum needs to be applied.
    """
    axis = (1, 2, 3)
    if tf.is_tensor(y_true) or isinstance(y_true, np.ndarray):
        if channel_average:
            axis = (1, 2)
    elif isinstance(y_true, torch.Tensor):
        if channel_average:
            axis = (2, 3)
    else:
        raise ValueError("Unsupported tensor type.")

    return axis


def dice_score(y_pred: Tensor,
               y_true: Tensor,
               soft_dice: bool = False,
               average_sample: bool = False,
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
    b = fe.backend.dice_score(y_pred=pred, y_true=true, average_sample=True)  # 0.161

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = tf.constant([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_score(y_pred=pred, y_true=true, average_sample=True)  # 0.161
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = torch.tensor([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # 0.161
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_score(y_pred=pred, y_true=true, average_sample=True)  # 0.161
    ```

    ```
    Args:
        y_pred: Prediction with a shape like (Batch, C, H, W) for torch and (Batch, H, W, C) for tensorflow or numpy. dtype: float32 or float16.
        y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
        soft_dice: Whether to square elements. If True, square of elements is added.
        average_sample: Whether to average the element-wise dice score.
        channel_average: Whether to average the channel wise dice score.

    Returns:
        The dice score between `y_pred` and `y_true`. A scalar if `average_sample` is True, else a
        tensor with the shape (Batch).

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types. if data type is other than np.array, tensor.Tensor, tf.Tensor.
    """
    y_true = convert_input_precision(y_true)
    y_pred = convert_input_precision(y_pred)
    epsilon = convert_input_precision(epsilon)

    axis = get_axis(y_true, channel_average)

    numerator = reduce_sum(y_true*y_pred, axis=axis)

    denominator = get_denominator(y_true, y_pred, soft_dice)

    denominator = reduce_sum(denominator, axis=axis)

    dice_score = (2 * (numerator + epsilon)) / (denominator + epsilon)

    if channel_average:
        dice_score = reduce_mean(dice_score, axis=1)

    if average_sample:
        dice_score = reduce_mean(dice_score)

    return dice_score
