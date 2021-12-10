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
from typing import TypeVar, Union

import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def l2_regularization(loss: Tensor, model: Union[tf.keras.Model, torch.nn.Module], beta: float = 0.01) -> Tensor:
    """Calculate mean squared error between two tensors.

    This method can be used with TensorFlow tensors:
    ```python
    true = tf.constant([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = tf.constant([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
    true = tf.constant([[1], [3], [2], [0]])
    pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
    ```

    This method can be used with PyTorch tensors:
    ```python
    true = torch.tensor([[0,1,0,0], [0,0,0,1], [0,0,1,0], [1,0,0,0]])
    pred = torch.tensor([[0.1,0.9,0.05,0.05], [0.1,0.2,0.0,0.7], [0.0,0.15,0.8,0.05], [1.0,0.0,0.0,0.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [0.0063, 0.035, 0.016, 0.0]
    true = tf.constant([[1], [3], [2], [0]])
    pred = tf.constant([[2.0], [0.0], [2.0], [1.0]])
    b = fe.backend.mean_squared_error(y_pred=pred, y_true=true)  # [1.0, 9.0, 0.0, 1.0]
    ```

    Args:
        y_true: Ground truth class labels with a shape like (batch) or (batch, n_classes). dtype: int, float16, float32.
        y_pred: Prediction score for each class, with a shape like y_true. dtype: float32 or float16.

    Returns:
        The MSE between `y_true` and `y_pred`

    Raises:
        AssertionError: If `y_true` and `y_pred` have mismatched shapes or disparate types.
        ValueError: If `y_pred` is an unacceptable data type.
    """
    if isinstance(model, torch.nn.Module):
        assert torch.is_tensor(loss), 'loss must bef=long to same framework as the model'
        l2_loss = torch.tensor(0.)
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += (torch.sum(param.pow(2))) / 2
        #total_loss = torch.add((beta * l2_loss), loss)

    elif isinstance(model, tf.keras.Model):
        assert tf.is_tensor(loss), 'loss must bef=long to same framework as the model'
        l2_loss = tf.zeros(1)[0]
        for layer in model.layers:
            for w in layer.trainable_variables:
                if tf.nn.l2_loss(w) != 0.0:
                    l2_loss += tf.nn.l2_loss(w)
        #total_loss = tf.add((beta * l2_loss)[0], loss)
    else:
        raise ValueError("Unrecognized model framework: Please make sure to pass either torch or tensorflow models")

    return beta * l2_loss