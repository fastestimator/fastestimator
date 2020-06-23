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


def sparse_categorical_crossentropy(y_pred: Tensor,
                                    y_true: Tensor,
                                    from_logits: bool = False,
                                    average_loss: bool = True) -> Tensor:
    """Compute sparse categorical crossentropy.

    Note that if any of the `y_pred` values are exactly 0, this will result in a NaN output. If `from_logits` is
    False, then each entry of `y_pred` should sum to 1. If they don't sum to 1 then tf and torch backends will
    result in different numerical values.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[1], [0], [2]])
    pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228
    b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.22, 0.11, 0.36]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[1], [0], [2]])
    pred = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true)  # 0.228
    b = fe.backend.sparse_categorical_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.22, 0.11, 0.36]
    ```

    Args:
        y_pred: Prediction with a shape like (Batch, C). dtype: float32 or float16.
        y_true: Ground truth class labels with a shape like (Batch) or (Batch, 1). dtype: int.
        from_logits: Whether y_pred is from logits. If True, a softmax will be applied to the prediction.
        average_loss: Whether to average the element-wise loss.

    Returns:
        The sparse categorical crossentropy between `y_pred` and `y_true`. A scalar if `average_loss` is True, else a
        tensor with the shape (Batch).

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """
    assert type(y_pred) == type(y_true), "y_pred and y_true must be same tensor type"
    assert isinstance(y_pred, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_pred"
    assert isinstance(y_true, (tf.Tensor, torch.Tensor)), "only support tf.Tensor or torch.Tensor as y_true"
    if tf.is_tensor(y_pred):
        ce = tf.losses.sparse_categorical_crossentropy(y_pred=y_pred, y_true=y_true, from_logits=from_logits)
    else:
        y_true = y_true.view(-1)
        if from_logits:
            ce = torch.nn.CrossEntropyLoss(reduction="none")(input=y_pred, target=y_true.long())
        else:
            ce = torch.nn.NLLLoss(reduction="none")(input=torch.log(y_pred), target=y_true.long())
    if average_loss:
        ce = reduce_mean(ce)
    return ce
