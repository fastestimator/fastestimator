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

from fastestimator.util.util import STRING_TO_TORCH_DTYPE

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def zeros_like(tensor: Tensor, dtype: Union[None, str] = None) -> Tensor:
    """Generate zeros shaped like `tensor` with a specified `dtype`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[0,1],[2,3]])
    b = fe.backend.zeros_like(n)  # [[0, 0], [0, 0]]
    b = fe.backend.zeros_like(n, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0,1],[2,3]])
    b = fe.backend.zeros_like(t)  # [[0, 0], [0, 0]]
    b = fe.backend.zeros_like(t, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0,1],[2,3]])
    b = fe.backend.zeros_like(p)  # [[0, 0], [0, 0]]
    b = fe.backend.zeros_like(p, dtype="float32")  # [[0.0, 0.0], [0.0, 0.0]]
    ```

    Args:
        tensor: The tensor whose shape will be copied.
        dtype: The data type to be used when generating the resulting tensor. If None then the `tensor` dtype is used.

    Returns:
        A tensor of zeros with the same shape as `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.zeros_like(tensor, dtype=dtype)
    elif isinstance(tensor, torch.Tensor):
        return torch.zeros_like(tensor, dtype=STRING_TO_TORCH_DTYPE[dtype])
    elif isinstance(tensor, np.ndarray):
        return np.zeros_like(tensor, dtype=dtype)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
