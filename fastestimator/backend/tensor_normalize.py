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

from fastestimator.backend import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def tensor_normalize(tensor: Tensor, mean, std, eps: float = 1e-6) -> Tensor:
    """Compute the mean value along a given `axis` of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_mean(n)  # 4.5
    b = fe.backend.reduce_mean(n, axis=0)  # [[3, 4], [5, 6]]
    b = fe.backend.reduce_mean(n, axis=1)  # [[2, 3], [6, 7]]
    b = fe.backend.reduce_mean(n, axis=[0,2])  # [3.5, 5.5]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_mean(t)  # 4.5
    b = fe.backend.reduce_mean(t, axis=0)  # [[3, 4], [5, 6]]
    b = fe.backend.reduce_mean(t, axis=1)  # [[2, 3], [3, 7]]
    b = fe.backend.reduce_mean(t, axis=[0,2])  # [3.5, 5.5]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reduce_mean(p)  # 4.5
    b = fe.backend.reduce_mean(p, axis=0)  # [[3, 4], [5, 6]]
    b = fe.backend.reduce_mean(p, axis=1)  # [[2, 3], [6, 7]]
    b = fe.backend.reduce_mean(p, axis=[0,2])  # [3.5, 5.5]
    ```

    Args:
        tensor: The input value.
        axis: Which axis or collection of axes to compute the mean along.
        keepdims: Whether to preserve the number of dimensions during the reduction.

    Returns:
        The mean values of `tensor` along `axis`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        mean = tf.convert_to_tensor(mean)
        std = tf.convert_to_tensor(std)
        return (tensor / 255 - mean) / std
    elif isinstance(tensor, torch.Tensor):
        mean = to_tensor(mean, "torch")
        std = to_tensor(std, "torch")
        tensor = (tensor / 255 - mean) / std
        return tensor.permute((0, 3, 1, 2))
    elif isinstance(tensor, np.ndarray):
        return (tensor / 255 - mean) / std
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
