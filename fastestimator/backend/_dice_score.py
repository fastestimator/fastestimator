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
from typing import List, Optional, TypeVar

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend._cast import cast
from fastestimator.backend._reduce_max import reduce_max
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._reduce_sum import reduce_sum
from fastestimator.backend._where import where

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def _get_channel_axis(tensor: Tensor) -> int:
    if tf.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return -1
    elif isinstance(tensor, torch.Tensor):
        return 1
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}.")


def _get_spacial_axes(tensor: Tensor, channel_axis: int) -> List[int]:
    dims = len(tensor.shape)
    if channel_axis == -1:
        return list(range(dims)[1:-1])
    return list(range(dims)[2:])


def dice_score(y_pred: Tensor,
               y_true: Tensor,
               soft_dice: bool = False,
               sample_average: bool = False,
               channel_average: bool = False,
               channel_weights: Optional[Tensor] = None,
               mutually_exclusive_channels: bool = False,
               threshold: Optional[float] = None,
               epsilon: float = 1e-6) -> Tensor:
    """Compute Dice score.

    This method can be used with Numpy data:
    ```python
    true = np.array([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                      [[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                      [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = np.array([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]],
                      [[0, 1, 1], [1, 0, 1], [0, 0, 0]],
                      [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # [[0.90909083, 0.79999984, 0.79999995]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # [[0.90909083, 0.79999984, 0.79999995]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # [0.83636354]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # [0.90909083, 0.79999984, 0.79999995]

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                         [[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                         [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = tf.constant([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]],
                         [[0, 1, 1], [1, 0, 1], [0, 0, 0]],
                         [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # [[0.9090908 , 0.79999983, 0.79999995]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # [[0.9090908 , 0.79999983, 0.79999995]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # [0.83636355]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # [0.9090908 , 0.79999983, 0.79999995]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[[[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                          [[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                          [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]])
    pred = torch.tensor([[[[0, 1, 0], [1, 0, 0], [1, 0, 1]],
                          [[0, 1, 1], [1, 0, 1], [0, 0, 0]],
                          [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]])
    b = fe.backend.dice_score(y_pred=pred, y_true=true)  # [[0.8000, 0.8000, 0.9091]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, soft_dice=True)  # [[0.8000, 0.8000, 0.9091]]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, channel_average=True)  # [0.8364]
    b = fe.backend.dice_score(y_pred=pred, y_true=true, sample_average=True)  # [0.8000, 0.8000, 0.9091]
    ```

    Args:
        y_pred: Prediction with a shape like (Batch, C, ...) for torch and (Batch, ..., C) for tensorflow or numpy.
        y_true: Ground truth class labels with a shape like `y_pred`.
        soft_dice: Whether to square elements in the denominator.
        sample_average: Whether to average the dice score along the batch dimension.
        channel_average: Whether to average the dice score along the channel dimension.
        channel_weights: A tensor of weights (size 1xN_Channels) to apply to each channel before reduction.
        mutually_exclusive_channels: Whether an individual pixel can belong to only 1 class (True) or more than 1 class
            (False). If True, a threshold of 0.0 can be used to binarize by whatever the maximum predicted class is.
        threshold: The threshold for binarizing the prediction. Set this to 0.0 if you are using a background class. Set
            to None if continuous values are desired (ex. for a loss).
        epsilon: floating point value to avoid divide by zero error.

    Returns:
        The dice score between `y_pred` and `y_true`.

    Raises:
        AssertionError: If `y_true` or `y_pred` something other than np.array, tensor.Tensor, or tf.Tensor.
    """
    y_true, y_pred = cast(y_true, 'float32'), cast(y_pred, 'float32')
    channel_axis = _get_channel_axis(y_pred)
    spacial_axes = _get_spacial_axes(y_pred, channel_axis=channel_axis)

    if mutually_exclusive_channels:
        # Find the max prediction per channel
        pick = reduce_max(y_pred, axis=channel_axis, keepdims=True)
        # Assign each pixel only to the max prediction across the channels
        y_pred = where(y_pred >= pick, y_pred, 0.0)
    if threshold is not None:
        # Only accept predictions which are over the given confidence threshold
        y_pred = where(y_pred > threshold, 1.0, 0.0)
        y_pred = cast(y_pred, 'float32')

    numerator = reduce_sum(y_pred * y_true, axis=spacial_axes)

    if soft_dice:
        denominator = reduce_sum(y_pred ** 2, axis=spacial_axes) + reduce_sum(y_true ** 2, axis=spacial_axes)
    else:
        denominator = reduce_sum(y_pred, axis=spacial_axes) + reduce_sum(y_true, axis=spacial_axes)

    dice = 2.0 * numerator / (denominator + epsilon)  # N x C

    if channel_weights is not None:
        channel_weights = cast(channel_weights, 'float32')
        dice = dice * channel_weights
    if channel_average:
        dice = reduce_mean(dice, axis=channel_axis)  # N
    if sample_average:
        dice = reduce_mean(dice, axis=0)  # 1 or C

    return dice
