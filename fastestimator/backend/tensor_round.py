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


def tensor_round(tensor: Tensor) -> Tensor:
    """Element-wise rounds the values of the `tensor` to nearest integer.

    This method can be used with Numpy data:
    ```python
    n = np.array([[1.25, 4.5, 6], [4, 9.11, 16]])
    b = fe.backend.tensor_round(n)  # [[1, 4, 6], [4, 9, 16]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[1.25, 4.5, 6], [4, 9.11, 16.9]])
    b = fe.backend.tensor_round(t)  # [[1, 4, 6], [4, 9, 17]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[1.25, 4.5, 6], [4, 9.11, 16]])
    b = fe.backend.tensor_round(p)  # [[1, 4, 6], [4, 9, 16]]
    ```

    Args:
        tensor: The input tensor.

    Returns:
        The rounded `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.round(tensor)
    elif isinstance(tensor, torch.Tensor):
        return torch.round(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.round(tensor)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
