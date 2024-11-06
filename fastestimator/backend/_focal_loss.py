# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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
import torch.nn.functional as F

from fastestimator.backend._clip_by_value import clip_by_value
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._reduce_sum import reduce_sum

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def pytorch_focal_loss(y_pred: torch.Tensor,
                       y_true: torch.Tensor,
                       alpha: float = 0.25,
                       gamma: float = 2,
                       from_logits: bool = False) -> torch.Tensor:
    """
    Calculate the focal loss between two tensors.

    Args:
        y_true: Ground truth class labels with shape([batch_size, d0, .. dN]), which should take values of 1 or 0.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.
        alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples or (-1/None) to ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        from_logits: Whether y_pred is logits (without sigmoid).
    Returns:
        Loss tensor.
    """
    if from_logits:
        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    else:
        p = y_pred
        ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")

    p_t = p * y_true + (1 - p) * (1 - y_true)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss
    return loss


def tf_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0):
    """Computes the binary focal crossentropy loss.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
    effect on the binary crossentropy loss.

    If `apply_class_balancing == True`, this function also takes into account a
    weight balancing factor for the binary classes 0 and 1 as follows:

    `weight = alpha` for class 1 (`target == 1`)
    `weight = 1 - alpha` for class 0
    where `alpha` is a float in the range of `[0, 1]`.

    Args:
        y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
        y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.

    Returns:
        Binary focal crossentropy loss value
        with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = tf_focal_loss(y_true, y_pred, gamma=2)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.330, 0.206], dtype=float32)
    """
    y_true = tf.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        y_pred = tf.math.sigmoid(y_pred)

    bce = tf.keras.backend.binary_crossentropy(
        y_true,
        y_pred,
        from_logits=False,
    )

    # Calculate focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = tf.math.pow(1.0 - p_t, gamma)

    focal_bce = focal_factor * bce

    if alpha >= 0:
        weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_bce = weight * focal_bce

    return focal_bce


def focal_loss(y_true: Tensor,
               y_pred: Tensor,
               gamma: float = 2.0,
               alpha: float = 0.25,
               from_logits: bool = False,
               normalize: bool = True,
               shape_reduction: str = "sum",
               sample_reduction: str = "mean",
               label_smoothing: float = 0.0) -> Tensor:
    """Calculate the focal loss between two tensors.

    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[1], [1], [1], [0], [0], [0]])
    pred = tf.constant([[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]])
    b = fe.backend.focal_loss(y_pred=pred, y_true=true, gamma=None, alpha=None) #0.1464
    b = fe.backend.focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25) #0.00395
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[1], [1], [1], [0], [0], [0]])
    pred = torch.tensor([[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]])
    b = fe.backend.focal_loss(y_pred=pred, y_true=true, gamma=None, alpha=None) #0.1464
    b = fe.backend.focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25) #0.004
    ```

    Args:
        y_true: Ground truth class labels with shape([batch_size, d0, .. dN]), which should take values of 1 or 0.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.
        alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples or (-1/None) to ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        normalize: Whether to normalize focal loss along samples based on number of positive classes per samples.
        shape_reduction:
                 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged across classes.
                 'sum': The output will be summed across classes.
        from_logits: Whether y_pred is logits (without sigmoid).
        sample_reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged across batch size.
                 'sum': The output will be summed across batch size.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
    Returns:
        The Focal loss between `y_true` and `y_pred`

    Raises:
        ValueError: If `y_pred` or 'y_true' is an unacceptable data type.
    """
    if gamma is None or gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    if alpha is None or (alpha < 0 or alpha > 1):
        raise ValueError("Value of alpha can either be -1 or None or within range (0, 1)")

    if tf.is_tensor(y_true):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        fl = tf_focal_loss(y_true,
                           y_pred,
                           from_logits=from_logits,
                           alpha=alpha,
                           gamma=gamma,
                           label_smoothing=label_smoothing)
        gt_shape = tf.shape(y_true)
        fl_shape = tf.shape(fl)
    elif isinstance(y_true, torch.Tensor):
        y_true = y_true.to(y_pred.dtype)
        fl = pytorch_focal_loss(y_pred=y_pred, y_true=y_true, alpha=alpha, gamma=gamma, from_logits=from_logits)
        gt_shape = y_true.shape
        fl_shape = fl.shape
    else:
        raise ValueError("Unsupported tensor type.")

    focal_reduce_axis = [*range(1, len(fl_shape))]
    # normalize along the batch size based on number of positive classes
    if normalize:
        gt_reduce_axis = [*range(1, len(gt_shape))]
        gt_count = clip_by_value(reduce_sum(y_true, axis=gt_reduce_axis), min_value=1)
        gt_count = gt_count[(..., ) + (None, ) * len(focal_reduce_axis)]
        fl = fl / gt_count

    if shape_reduction == "sum":
        fl = reduce_sum(fl, axis=focal_reduce_axis)
    elif shape_reduction == "mean":
        fl = reduce_mean(fl, axis=focal_reduce_axis)

    if sample_reduction == "mean":
        fl = reduce_mean(fl)
    elif sample_reduction == "sum":
        fl = reduce_sum(fl)

    return fl
