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


def binary_crossentropy(y_pred: Tensor, y_true: Tensor, from_logits: bool = False, average_loss: bool = True) -> Tensor:
    """Compute binary crossentropy.

    This method is applicable when there are only two label classes (zero and one). There should be a single floating
    point prediction per example.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[1], [0], [1], [0]])
    pred = tf.constant([[0.9], [0.3], [0.8], [0.1]])
    b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197
    b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[1], [0], [1], [0]])
    pred = torch.tensor([[0.9], [0.3], [0.8], [0.1]])
    b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true)  # 0.197
    b = fe.backend.binary_crossentropy(y_pred=pred, y_true=true, average_loss=False)  # [0.105, 0.356, 0.223, 0.105]
    ```

    Args:
        y_pred: Prediction with a shape like (batch, ...). dtype: float32 or float16.
        y_true: Ground truth class labels with the same shape as `y_pred`. dtype: int or float32 or float16.
        from_logits: Whether y_pred is from logits. If True, a sigmoid will be applied to the prediction.
        average_loss: Whether to average the element-wise loss.

    Returns:
        The binary crossentropy between `y_pred` and `y_true`. A scalar if `average_loss` is True, else a tensor with
        the same shape as `y_true`.

    Raises:
        AssertionError: If `y_true` or `y_pred` are unacceptable data types.
    """
    assert type(y_pred) is type(y_true), "y_pred and y_true must be same tensor type"
    assert isinstance(y_pred, torch.Tensor) or tf.is_tensor(y_pred), "only support tf.Tensor or torch.Tensor as y_pred"
    assert isinstance(y_true, torch.Tensor) or tf.is_tensor(y_true), "only support tf.Tensor or torch.Tensor as y_true"
    if tf.is_tensor(y_pred):
        ce = tf.losses.binary_crossentropy(y_pred=y_pred,
                                           y_true=tf.reshape(y_true, tf.shape(y_pred)),
                                           from_logits=from_logits)
        ce = tf.reshape(ce, [tf.shape(ce)[0], -1])
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
        ce = reduce_mean(ce)
    return ce
