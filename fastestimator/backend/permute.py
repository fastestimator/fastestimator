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
from typing import TypeVar, List

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def permute(tensor: Tensor, permutation: List[int]) -> Tensor:
    """Perform the specified `permutation` on the axes of a given `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.permute(n, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]
    b = fe.backend.permute(n, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.permute(t, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]
    b = fe.backend.permute(t, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.permute(p, [2, 0, 1])  # [[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]]
    b = fe.backend.permute(P, [0, 2, 1])  # [[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]]
    ```

    Args:
        tensor: The tensor to permute.
        permutation: The new axis order to be used. Should be a list containing all integers in range [0, tensor.ndim).

    Returns:
        The `tensor` with axes swapped according to the `permutation`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.transpose(tensor, perm=permutation)
    elif isinstance(tensor, torch.Tensor):
        return tensor.permute(*permutation)
    elif isinstance(tensor, np.ndarray):
        return np.transpose(tensor, axes=permutation)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
