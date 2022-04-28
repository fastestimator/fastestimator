# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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


def flip(tensor: Tensor, axis: List[int]) -> Tensor:
    """Reverse the order of a given `tensor` elements along a given axis.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.flip(n,axis = [0])  # [[[8,  9], [10, 11]], [[4,  5], [6,  7]], [[0,  1], [2,  3]]]
    b = fe.backend.flip(n,axis = [0,1])  # [[[10, 11],[8,  9]], [[6,  7], [4,  5]], [[2,  3], [0,  1]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.flip(t,axis = [0])  # [[[8,  9], [10, 11]], [[4,  5], [6,  7]], [[0,  1], [2,  3]]]
    b = fe.backend.flip(t,axis = [0,1])  # [[[10, 11],[8,  9]], [[6,  7], [4,  5]], [[2,  3], [0,  1]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.flip(p,axis = [0])  # [[[8,  9], [10, 11]], [[4,  5], [6,  7]], [[0,  1], [2,  3]]]
    b = fe.backend.flip(p,axis = [0,1])  # [[[10, 11],[8,  9]], [[6,  7], [4,  5]], [[2,  3], [0,  1]]]
    ```

    Args:
        tensor: The tensor to flip.
        axis: The new axis order to be used. Should be a list containing all integers in range [0, tensor.ndim).

    Returns:
        The `tensor` with axes flipped according to the `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.reverse(tensor, axis=axis)
    elif isinstance(tensor, torch.Tensor):
        return torch.flip(tensor, dims=axis)
    elif isinstance(tensor, np.ndarray):
        return np.flip(tensor, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
