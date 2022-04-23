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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def tensor_pow(tensor: Tensor, power: Union[int, float]) -> Tensor:
    """Computes x^power element-wise along `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[1, 4, 6], [2.3, 0.5, 0]])
    b = fe.backend.tensor_pow(n, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
    b = fe.backend.tensor_pow(n, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[1, 4, 6], [2.3, 0.5, 0]])
    b = fe.backend.tensor_pow(t, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
    b = fe.backend.tensor_pow(t, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[1, 4, 6], [2.3, 0.5, 0]])
    b = fe.backend.tensor_pow(p, 3.2)  # [[1.0, 84.449, 309.089], [14.372, 0.109, 0]]
    b = fe.backend.tensor_pow(p, 0.21)  # [[1.0, 1.338, 1.457], [1.191, 0.865, 0]]
    ```

    Args:
        tensor: The input tensor.
        power: The power to which to raise the elements in the `tensor`.

    Returns:
        The `tensor` raised element-wise to the given `power`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.pow(tensor, power)
    elif isinstance(tensor, torch.Tensor):
        return torch.pow(tensor, power)
    elif isinstance(tensor, np.ndarray):
        return np.power(tensor, power)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
