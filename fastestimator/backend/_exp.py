# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def exp(tensor: Tensor) -> Tensor:
    """Compute e^Tensor.

    This method can be used with Numpy data:
    ```python
    n = np.array([-2, 2, 1])
    b = fe.backend.exp(n)  # [0.1353, 7.3891, 2.7183]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([-2.0, 2, 1])
    b = fe.backend.exp(t)  # [0.1353, 7.3891, 2.7183]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([-2.0, 2, 1])
    b = fe.backend.exp(p)  # [0.1353, 7.3891, 2.7183]
    ```

    Args:
        tensor: The input value.

    Returns:
        The exponentiated `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.exp(tensor)
    elif isinstance(tensor, torch.Tensor):
        return torch.exp(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.exp(tensor)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
