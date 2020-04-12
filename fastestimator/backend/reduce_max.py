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

from typing import TypeVar, Union, Sequence

import numpy as np
import tensorflow as tf
import torch

from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def reduce_max(tensor: Tensor, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False) -> Tensor:
    """Compute the maximum value along a given `axis` of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = fe.backend.reduce_max(n)  # 8
    b = fe.backend.reduce_max(n, axis=0)  # [[5, 6], [7, 8]]
    b = fe.backend.reduce_max(n, axis=1)  # [[3, 4], [7, 8]]
    b = fe.backend.reduce_max(n, axis=[0,2])  # [6, 8]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = fe.backend.reduce_max(t)  # 8
    b = fe.backend.reduce_max(t, axis=0)  # [[5, 6], [7, 8]]
    b = fe.backend.reduce_max(t, axis=1)  # [[3, 4], [7, 8]]
    b = fe.backend.reduce_max(t, axis=[0,2])  # [6, 8]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = fe.backend.reduce_max(p)  # 8
    b = fe.backend.reduce_max(p, axis=0)  # [[5, 6], [7, 8]]
    b = fe.backend.reduce_max(p, axis=1)  # [[3, 4], [7, 8]]
    b = fe.backend.reduce_max(p, axis=[0,2])  # [6, 8]
    ```

    Args:
        tensor: The input value.
        axis: Which axis or collection of axes to compute the maximum along.
        keepdims: Whether to preserve the number of dimensions during the reduction.

    Returns:
        The maximum values of `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.reduce_max(tensor, axis=axis, keepdims=keepdims)
    elif isinstance(tensor, torch.Tensor):
        if axis is None:
            axis = list(range(len(tensor.shape)))
        axis = to_list(axis)
        axis = reversed(sorted(axis))
        for ax in axis:
            tensor = tensor.max(dim=ax, keepdim=keepdims)[0]
        return tensor
    elif isinstance(tensor, np.ndarray):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.max(tensor, axis=axis, keepdims=keepdims)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
