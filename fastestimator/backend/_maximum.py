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


def maximum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Get the maximum of the given `tensors`.

    This method can be used with Numpy data:
    ```python
    n1 = np.array([[2, 7, 6]])
    n2 = np.array([[2, 7, 5]])
    res = fe.backend.maximum(n1, n2) # [[2, 7, 6]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t1 = tf.constant([[2, 7, 6]])
    t2 = tf.constant([[2, 7, 5]])
    res = fe.backend.maximum(t1, t2) # [[2, 7, 6]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p1 = torch.tensor([[2, 7, 6]])
    p2 = torch.tensor([[2, 7, 5]])
    res = fe.backend.maximum(p1, p2) # [[2, 7, 6]]
    ```

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.

    Returns:
        The maximum of two `tensors`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor1) and tf.is_tensor(tensor2):
        return tf.maximum(tensor1, tensor2)
    elif isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        return torch.max(tensor1, tensor2)
    elif isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray):
        return np.maximum(tensor1, tensor2)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor1)))
