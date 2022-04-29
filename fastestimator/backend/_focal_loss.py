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

from fastestimator.backend._cast import cast
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._reduce_sum import reduce_sum

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def focal_loss(y_true: Tensor, y_pred: Tensor, gamma: float = 2.0, alpha: float = 0.25,  from_logits: bool = False, reduction: str = "mean") -> Tensor:
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
        y_true: Ground truth class labels which should take values of 1 or -1.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.
        alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples or (-1/None) to ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        from_logits: Whether y_pred is logits (without softmax).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The Focal loss between `y_true` and `y_pred`

    Raises:
        ValueError: If `y_pred` or 'y_true' is an unacceptable data type.
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero.")

    if alpha:
        if alpha != -1 and (alpha < 0 or alpha > 1):
            raise ValueError(
                "Value of alpha can either be -1 or None or within range (0, 1)")

    if alpha and alpha != -1:
        alpha = cast(alpha, y_pred)

    if gamma:
        gamma = cast(gamma, y_pred)

    if tf.is_tensor(y_true):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        ce = tf.keras.losses.binary_crossentropy(y_pred=y_pred,
                                                 y_true=y_true,
                                                 from_logits=from_logits)
        ce = tf.reshape(ce, [tf.shape(ce)[0], -1])

        # If logits are provided then convert the predictions into probabilities
        if from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred

    elif isinstance(y_true, torch.Tensor):
        y_true = y_true.to(y_pred.dtype)

        if from_logits:
            ce = torch.nn.BCEWithLogitsLoss(reduction="none")(
                input=y_pred, target=y_true)
            pred_prob = torch.sigmoid(y_pred)
        else:
            ce = torch.nn.BCELoss(reduction="none")(
                input=y_pred, target=y_true)
            pred_prob = y_pred
        ce = ce.view(ce.shape[0], -1)
    else:
        raise ValueError("Unsupported tensor type.")

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))

    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha and alpha != -1:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        modulating_factor = (1.0 - p_t)**gamma

    loss = alpha_factor * modulating_factor * ce
    if reduction == "mean":
        loss = reduce_mean(loss)
    elif reduction == "sum":
        loss = reduce_sum(loss)

    return loss
