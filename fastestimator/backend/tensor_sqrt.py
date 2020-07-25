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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def tensor_sqrt(tensor: Tensor) -> Tensor:
    """Computes element-wise square root of tensor elements.

    This method can be used with Numpy data:
    ```python
    n = np.array([[1, 4, 6], [4, 9, 16]])
    b = fe.backend.tensor_sqrt(n)  # [[1.0, 2.0, 2.44948974], [2.0, 3.0, 4.0]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[1, 4, 6], [4, 9, 16]], dtype=tf.float32)
    b = fe.backend.tensor_sqrt(t)  # [[1.0, 2.0, 2.4494898], [2.0, 3.0, 4.0]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[1, 4, 6], [4, 9, 16]], dtype=torch.float32)
    b = fe.backend.tensor_sqrt(p)  # [[1.0, 2.0, 2.4495], [2.0, 3.0, 4.0]]
    ```

    Args:
        tensor: The input tensor.

    Returns:
        The `tensor` that contains square root of input values.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.sqrt(tensor)
    elif isinstance(tensor, torch.Tensor):
        return torch.sqrt(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.sqrt(tensor)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
