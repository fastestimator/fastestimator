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

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.reduce_mean import reduce_mean

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def reduce_loss(loss: Tensor) -> Tensor:
    """Compute the mean value of a 1D `loss`, while leaving scalar `loss` inputs unmodified.

    This method can be used with Numpy data:
    ```python
    n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = fe.backend.reduce_loss(n)  # 3.0
    n = np.array(6.0)
    b = fe.backend.reduce_loss(n)  # 6.0
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    b = fe.backend.reduce_loss(t)  # 3
    t = tf.constant(6.0)
    b = fe.backend.reduce_loss(t)  # 6.0
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = fe.backend.reduce_loss(p)  # [[-0.6, 0.2], [1.9, -0.02]]
    p = torch.tensor(6.0)
    b = fe.backend.reduce_loss(p)  # 6.0
    ```

    Args:
        loss: A tensor to be reduced. Should have a floating point dtype.

    Returns:
        The mean value of the `loss`.

    Raises:
        AssertionError: If `loss` is has more than one dimension.
    """
    assert len(loss.shape) < 2, "loss must be one-dimensional or scalar"
    if len(loss.shape) == 1:
        loss = reduce_mean(loss)
    return loss
