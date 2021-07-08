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

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def feed_forward(model: Union[tf.keras.Model, torch.nn.Module], *x: Union[Tensor, np.ndarray],
                 training: bool = True) -> Tensor:
    """Run a forward step on a given model.

    This method can be used with TensorFlow models:
    ```python
    m = fe.architecture.tensorflow.LeNet(classes=2)
    x = tf.ones((3,28,28,1))  # (batch, height, width, channels)
    b = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    ```

    This method can be used with PyTorch models:
    ```python
    m = fe.architecture.pytorch.LeNet(classes=2)
    x = torch.ones((3,1,28,28))  # (batch, channels, height, width)
    b = fe.backend.feed_forward(m, x)  # [[~0.5, ~0.5], [~0.5, ~0.5], [~0.5, ~0.5]]
    ```

    Args:
        model: A neural network to run the forward step through.
        x: One or more input tensor for the `model`. This value will be auto-cast to either a tf.Tensor or torch.Tensor
            as applicable for the `model`.
        training: Whether this forward step is part of training or not. This may impact the behavior of `model` layers
            such as dropout.

    Returns:
        The result of `model(x)`.

    Raises:
        ValueError: If `model` is an unacceptable data type.
    """
    if isinstance(model, tf.keras.Model):
        x = to_tensor(x, "tf")
        x = model(*x, training=training)
    elif isinstance(model, torch.nn.Module):
        model.train(mode=training)
        x = to_tensor(x, "torch")
        x = model(*x)
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
    return x
