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

from fastestimator.backend.reduce_loss import reduce_loss

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def binary_crossentropy(y_pred: Tensor, y_true: Tensor, from_logits: bool = False, average_loss: bool = True) -> Tensor:
    """calculate binary crossentropy

    Args:
        y_pred: prediction with any shape dtype: float32
        y_true: ground truth with shape same as prediction, dtype: int or float32
        from_logits: whether y_pred is from logits, if yes, sigmoid will be applied to prediction. Defaults to False.
        average_loss: whether to average the element-wise loss

    Returns:
        Tensor: binary cross entropy
    """
    assert type(y_pred) is type(y_true), "y_pred and y_true must be same tensor type"
    assert isinstance(y_pred, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_pred"
    assert isinstance(y_true, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_true"
    if isinstance(y_pred, tf.Tensor):
        ce = tf.losses.binary_crossentropy(y_pred=y_pred,
                                           y_true=tf.reshape(y_true, y_pred.shape),
                                           from_logits=from_logits)
        ce = tf.reshape(ce, [ce.shape[0], -1])
        ce = tf.reduce_mean(ce, 1)
    else:
        y_true = y_true.to(torch.float)
        if from_logits:
            ce = torch.nn.BCEWithLogitsLoss(reduction="none")(input=y_pred, target=y_true.view(y_pred.size()))
        else:
            ce = torch.nn.BCELoss(reduction="none")(input=y_pred, target=y_true.view(y_pred.size()))
        ce = ce.view(ce.shape[0], -1)
        ce = torch.mean(ce, dim=1)

    if average_loss:
        ce = reduce_loss(ce)
    return ce
