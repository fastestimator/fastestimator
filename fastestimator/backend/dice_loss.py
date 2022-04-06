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

from fastestimator.backend.clip_by_value import clip_by_value
from fastestimator.backend.reduce_mean import reduce_mean
from fastestimator.backend.reduce_sum import reduce_sum

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def get_loss(dice: Tensor, log_loss: bool, eps: float = 1e-6) -> Tensor:
    """
        Calculate log loss/dice loss

        Args:
            dice: The dice score
            log_loss: Whether to calculate log loss or not
            eps: Constant to avoid nan error

        Returns:
            Dice loss or log loss

    """
    dice = clip_by_value(dice, eps)
    if log_loss:
        if tf.is_tensor(dice):
            return -tf.math.log(dice)
        elif isinstance(dice, torch.Tensor):
            return -torch.log(dice)
        elif isinstance(dice, np.ndarray):
            return -np.log(dice)
    else:
        return 1.0 - dice


def get_soft_dice(y_true: Tensor, y_pred: Tensor, soft_dice_loss: bool) -> Tensor:
    """
        Calculate sum/squared sum of y_true and y_pred

        Args:
            y_pred: Prediction with a shape like (Batch, C). dtype: float32 or float16.
            y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
            soft_dice_loss: Whether to add direct sum or square sum of inputs

        Return:
            The sum or squared sum of y_pred and y_true
    """
    if soft_dice_loss:
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
            channel_average: Whether to average the channel wise loss.

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


def dice_loss(y_pred: Tensor,
              y_true: Tensor,
              log_loss: bool = False,
              soft_dice_loss: bool = False,
              average_sample_loss: bool = False,
              channel_average: bool = False) -> Tensor:
    """

    Compute Dice Loss/log loss.

    log_loss = -log(dice_score)

    dice_loss = 1 - dice_score

    This method can be used with Numpy data:
    ```python
    true = np.array([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = np.array([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_loss(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, soft_dice_loss=True)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, log_loss=True)  # 0.176
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, average_sample_loss=True)  # 0.161

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = tf.constant([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_loss(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, soft_dice_loss=True)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, log_loss=True)  # 0.176
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, average_sample_loss=True)  # 0.161
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = torch.tensor([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1], [0, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_loss(y_pred=pred, y_true=true)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, soft_dice_loss=True)  # 0.161
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, log_loss=True)  # 0.176
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, channel_average=True)  # 0.1636
    b = fe.backend.dice_loss(y_pred=pred, y_true=true, average_sample_loss=True)  # 0.161
    ```

    ```
    Args:
        y_pred: Prediction with a shape like (Batch, C). dtype: float32 or float16.
        y_true: Ground truth class labels with a shape like `y_pred`. dtype: int or float32 or float16.
        log_loss: Whether to commute log loss. If True, log loss will be applied instead of dice loss.
        soft_dice_loss: Whether to square elements. If True, square of elements is added.
        average_sample_loss: Whether to average the element-wise loss.
        channel_average: Whether to average the channel wise loss.

    Returns:
        The dice loss between `y_pred` and `y_true`. A scalar if `average_sample_loss` is True, else a
        tensor with the shape (Batch).

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """

    axis = get_axis(y_true, channel_average)

    numerator = reduce_sum(y_true*y_pred, axis=axis)

    denominator = get_soft_dice(y_true, y_pred, soft_dice_loss)

    denominator = reduce_sum(denominator, axis=axis)

    denominator = clip_by_value(denominator, 1)

    dice = (2 * numerator) / denominator

    if channel_average:
        dice = reduce_mean(dice, axis=1)

    loss = get_loss(dice, log_loss)

    if average_sample_loss:
        loss = reduce_mean(loss)

    return loss
