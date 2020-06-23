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


def argmax(tensor: Tensor, axis: int = 0) -> Tensor:
    """Compute the index of the maximum value along a given axis of a tensor.

    This method can be used with Numpy data:
    ```python
    n = np.array([[2,7,5],[9,1,3],[4,8,2]])
    b = fe.backend.argmax(n, axis=0)  # [1, 2, 0]
    b = fe.backend.argmax(n, axis=1)  # [1, 0, 1]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[2,7,5],[9,1,3],[4,8,2]])
    b = fe.backend.argmax(t, axis=0)  # [1, 2, 0]
    b = fe.backend.argmax(t, axis=1)  # [1, 0, 1]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[2,7,5],[9,1,3],[4,8,2]])
    b = fe.backend.argmax(p, axis=0)  # [1, 2, 0]
    b = fe.backend.argmax(p, axis=1)  # [1, 0, 1]
    ```

    Args:
        tensor: The input value.
        axis: Which axis to compute the index along.

    Returns:
        The indices corresponding to the maximum values within `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.argmax(tensor, axis=axis)
    elif isinstance(tensor, torch.Tensor):
        return tensor.max(dim=axis, keepdim=False)[1]
    elif isinstance(tensor, np.ndarray):
        return np.argmax(tensor, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
