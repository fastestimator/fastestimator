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
from typing import Sequence, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def reduce_sum(tensor: Tensor, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False) -> Tensor:
    """Compute the sum along a given `axis` of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_sum(n)  # 36
    b = fe.backend.reduce_sum(n, axis=0)  # [[6, 8], [10, 12]]
    b = fe.backend.reduce_sum(n, axis=1)  # [[4, 6], [12, 14]]
    b = fe.backend.reduce_sum(n, axis=[0,2])  # [14, 22]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_sum(t)  # 36
    b = fe.backend.reduce_sum(t, axis=0)  # [[6, 8], [10, 12]]
    b = fe.backend.reduce_sum(t, axis=1)  # [[4, 6], [12, 14]]
    b = fe.backend.reduce_sum(t, axis=[0,2])  # [14, 22]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_sum(p)  # 36
    b = fe.backend.reduce_sum(p, axis=0)  # [[6, 8], [10, 12]]
    b = fe.backend.reduce_sum(p, axis=1)  # [[4, 6], [12, 14]]
    b = fe.backend.reduce_sum(p, axis=[0,2])  # [14, 22]
    ```

    Args:
        tensor: The input value.
        axis: Which axis or collection of axes to compute the sum along.
        keepdims: Whether to preserve the number of dimensions during the reduction.

    Returns:
        The sum of `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.reduce_sum(tensor, axis=axis, keepdims=keepdims)
    elif isinstance(tensor, torch.Tensor):
        if axis is None:
            axis = list(range(len(tensor.shape)))
        return tensor.sum(dim=axis, keepdim=keepdims)
    elif isinstance(tensor, np.ndarray):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.sum(tensor, axis=axis, keepdims=keepdims)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
