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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def cross_entropy(y_pred: Tensor, y_true: Tensor, apply_softmax: bool = False) -> Tensor:
    """calculate cross entropy for tensor inputs

    Args:
        y_pred: prediction score for each class, in [Batch, C]
        y_true: ground truth class label index, in [Batch]
        apply_softmax: whether to apply softmax to y_pred. Defaults to False.

    Returns:
        categorical cross entropy
    """
    assert type(y_pred) == type(y_true), "y_pred and y_true must be same tensor type"
    assert isinstance(y_pred, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_pred"
    assert isinstance(y_true, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_true"
    if isinstance(y_pred, tf.Tensor):
        ce = tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=apply_softmax)
    else:
        if apply_softmax:
            ce = torch.nn.CrossEntropyLoss(reduction="none")(y_pred, y_true)
        else:
            ce = torch.nn.NLLLoss(reduction="none")(torch.log(y_pred), y_true.long())
    return ce
