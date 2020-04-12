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

from fastestimator.backend.expand_dims import expand_dims
from fastestimator.backend.squeeze import squeeze
from fastestimator.backend.to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def gather_from_batch(tensor: Tensor, indices: Tensor) -> Tensor:
    """Gather specific indices from a batch of data.

    This method can be useful if you need to compute gradients based on a specific subset of a tensor's output values.
    The `indices` will automatically be cast to the correct type (tf, torch, np) based on the type of the `tensor`.

    This method can be used with Numpy data:
    ```python
    ind = np.array([1, 0, 1])
    n = np.array([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather_from_batch(n, ind)  # [1, 2, 5]
    n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather_from_batch(n, ind)  # [[2, 3], [4, 5], [10, 11]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    ind = tf.constant([1, 0, 1])
    t = tf.constant([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather_from_batch(t, ind)  # [1, 2, 5]
    t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather_from_batch(t, ind)  # [[2, 3], [4, 5], [10, 11]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    ind = torch.tensor([1, 0, 1])
    p = torch.tensor([[0, 1], [2, 3], [4, 5]])
    b = fe.backend.gather_from_batch(p, ind)  # [1, 2, 5]
    p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.gather_from_batch(p, ind)  # [[2, 3], [4, 5], [10, 11]]
    ```

    Args:
        tensor: A tensor of shape (batch, d1, ..., dn).
        indices: A tensor of shape (batch, ) or (batch, 1) indicating which indices should be selected.

    Returns:
        A tensor of shape (batch, d2, ..., dn) containing the elements from `tensor` at the given `indices`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        indices = to_tensor(indices, 'tf')
        indices = tf.cast(indices, tf.int64)
        if len(indices.shape) == 1:  # Indices not batched
            indices = expand_dims(indices, 1)
        return tf.gather_nd(tensor, indices=indices, batch_dims=1)
    elif isinstance(tensor, torch.Tensor):
        return tensor[torch.arange(tensor.shape[0]), squeeze(indices)]
    elif isinstance(tensor, np.ndarray):
        return tensor[np.arange(tensor.shape[0]), squeeze(indices)]
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
