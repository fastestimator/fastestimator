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
from typing import List, TypeVar

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def reshape(tensor: Tensor, shape: List[int]) -> Tensor:
    """Reshape a `tensor` to conform to a given shape.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(n, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(n, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(n, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(n, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(t, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(t, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(t, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(t, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(p, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(p, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(p, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(p, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    Args:
        tensor: The input value.
        shape: The new shape of the tensor. At most one value may be -1 which indicates that whatever values are left
            should be packed into that axis.

    Returns:
        The reshaped `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.reshape(tensor, shape=shape)
    elif isinstance(tensor, torch.Tensor):
        return torch.reshape(tensor, shape=shape)
    elif isinstance(tensor, np.ndarray):
        return np.reshape(tensor, shape)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
