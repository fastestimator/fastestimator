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
from typing import Sequence, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def reduce_std(tensor: Tensor, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False) -> Tensor:
    """Compute the std value along a given `axis` of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(n)  # 2.2913
    b = fe.backend.reduce_std(n, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(n, axis=1)  # [[1., 1.], [1., 1.]]
    b = fe.backend.reduce_std(n, axis=[0,2])  # [2.23606798 2.23606798]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(t)  # 2.2913
    b = fe.backend.reduce_std(t, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(t, axis=1)  # [[2, 3], [3, 7]]
    b = fe.backend.reduce_std(t, axis=[0,2])  # [2.23606798 2.23606798]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_std(p)  # 2.2913
    b = fe.backend.reduce_std(p, axis=0)  # [[2., 2.], [2., 2.]]
    b = fe.backend.reduce_std(p, axis=1)  # [[1., 1.], [1., 1.]]
    b = fe.backend.reduce_std(p, axis=[0,2])  # [2.23606798 2.23606798]
    ```

    Args:
        tensor: The input value.
        axis: Which axis or collection of axes to compute the std along.
        keepdims: Whether to preserve the number of dimensions during the reduction.

    Returns:
        The std values of `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.math.reduce_std(tensor, axis=axis, keepdims=keepdims)
    elif isinstance(tensor, torch.Tensor):
        if axis is None:
            if not keepdims:
                return tensor.std(unbiased=False)
            axis = list(range(len(tensor.shape)))
        tensor = tensor.std(dim=axis, unbiased=False, keepdim=keepdims)
        return tensor
    elif isinstance(tensor, np.ndarray):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.std(tensor, axis=axis, keepdims=keepdims)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
