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

from fastestimator.backend._squeeze import squeeze
from fastestimator.backend._to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def gather(tensor: Tensor, indices: Tensor) -> Tensor:
    """Gather specific indices from a tensor.

    The `indices` will automatically be cast to the correct type (tf, torch, np) based on the type of the `tensor`.

    This method can be used with Numpy data:
    ```python
    ind = np.array([1, 0, 1])
    n = np.array([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather(n, ind)  # [[2, 3], [0, 1], [2, 3]]
    n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather(n, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    ind = tf.constant([1, 0, 1])
    t = tf.constant([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather(t, ind)  # [[2, 3], [0, 1], [2, 3]]
    t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather(t, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    ind = torch.tensor([1, 0, 1])
    p = torch.tensor([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather(p, ind)  # [[2, 3], [0, 1], [2, 3]]
    p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather(p, ind)  # [[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    ```

    Args:
        tensor: A tensor to gather values from.
        indices: A tensor indicating which indices should be selected. These represent locations along the 0 axis.

    Returns:
        A tensor containing the elements from `tensor` at the given `indices`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        indices = to_tensor(indices, 'tf')
        indices = tf.cast(indices, tf.int64)
        return tf.gather(tensor, indices=squeeze(indices), axis=0)
    elif isinstance(tensor, torch.Tensor):
        return tensor[squeeze(indices).type(torch.int64)]
    elif isinstance(tensor, np.ndarray):
        return np.take(tensor, squeeze(indices).astype('int64'), axis=0)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
