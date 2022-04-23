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
from typing import TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.util.util import STRING_TO_TORCH_DTYPE

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def random_normal_like(tensor: Tensor, mean: float = 0.0, std: float = 1.0,
                       dtype: Union[None, str] = 'float32') -> Tensor:
    """Generate noise shaped like `tensor` from a random normal distribution with a given `mean` and `std`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[0,1],[2,3]])
    b = fe.backend.random_normal_like(n)  # [[-0.6, 0.2], [1.9, -0.02]]
    b = fe.backend.random_normal_like(n, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0,1],[2,3]])
    b = fe.backend.random_normal_like(t)  # [[-0.6, 0.2], [1.9, -0.02]]
    b = fe.backend.random_normal_like(t, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0,1],[2,3]])
    b = fe.backend.random_normal_like(p)  # [[-0.6, 0.2], [1.9, -0.02]]
    b = fe.backend.random_normal_like(P, mean=5.0)  # [[3.7, 5.7], [5.6, 3.6]]
    ```

    Args:
        tensor: The tensor whose shape will be copied.
        mean: The mean of the normal distribution to be sampled.
        std: The standard deviation of the normal distribution to be sampled.
        dtype: The data type to be used when generating the resulting tensor. This should be one of the floating point
            types.

    Returns:
        A tensor of random normal noise with the same shape as `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.random.normal(shape=tensor.shape, mean=mean, stddev=std, dtype=dtype)
    elif isinstance(tensor, torch.Tensor):
        return torch.randn_like(tensor, dtype=STRING_TO_TORCH_DTYPE[dtype]) * std + mean
    elif isinstance(tensor, np.ndarray):
        return np.random.normal(loc=mean, scale=std, size=tensor.shape).astype(dtype=dtype)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
