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


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Perform matrix multiplication on `a` and `b`.

    This method can be used with Numpy data:
    ```python
    a = np.array([[0,1,2],[3,4,5]])
    b = np.array([[1],[2],[3]])
    c = fe.backend.matmul(a, b)  # [[8], [26]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    a = tf.constant([[0,1,2],[3,4,5]])
    b = tf.constant([[1],[2],[3]])
    c = fe.backend.matmul(a, b)  # [[8], [26]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    a = torch.tensor([[0,1,2],[3,4,5]])
    b = torch.tensor([[1],[2],[3]])
    c = fe.backend.matmul(a, b)  # [[8], [26]]
    ```

    Args:
        a: The first matrix.
        b: The second matrix.

    Returns:
        The matrix multiplication result of a * b.

    Raises:
        ValueError: If either `a` or `b` are unacceptable or non-matching data types.
    """
    if tf.is_tensor(a) and tf.is_tensor(b):
        return tf.matmul(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.matmul(b)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.matmul(a, b)
    elif type(a) != type(b):
        raise ValueError(f"Tensor types do not match ({type(a)} and {type(b)})")
    else:
        raise ValueError(f"Unrecognized tensor type ({type(a)} or {type(b)})")
