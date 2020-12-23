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


def random_uniform_like(tensor: Tensor, minval: float = 0.0, maxval: float = 1.0,
                        dtype: Union[None, str] = 'float32') -> Tensor:
    """Generate noise shaped like `tensor` from a random normal distribution with a given `mean` and `std`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[0,1],[2,3]])
    b = fe.backend.random_uniform_like(n)  # [[0.62, 0.49], [0.88, 0.37]]
    b = fe.backend.random_uniform_like(n, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0,1],[2,3]])
    b = fe.backend.random_uniform_like(t)  # [[0.62, 0.49], [0.88, 0.37]]
    b = fe.backend.random_uniform_like(t, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0,1],[2,3]])
    b = fe.backend.random_uniform_like(p)  # [[0.62, 0.49], [0.88, 0.37]]
    b = fe.backend.random_uniform_like(P, minval=-5.0, maxval=-3)  # [[-3.8, -4.4], [-4.8, -4.9]]
    ```

    Args:
        tensor: The tensor whose shape will be copied.
        minval: The minimum bound of the uniform distribution.
        maxval: The maximum bound of the uniform distribution.
        dtype: The data type to be used when generating the resulting tensor. This should be one of the floating point
            types.

    Returns:
        A tensor of random uniform noise with the same shape as `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.random.uniform(shape=tensor.shape, minval=minval, maxval=maxval, dtype=dtype)
    elif isinstance(tensor, torch.Tensor):
        return torch.rand_like(tensor, dtype=STRING_TO_TORCH_DTYPE[dtype]) * (maxval - minval) + minval
    elif isinstance(tensor, np.ndarray):
        return np.random.uniform(low=minval, high=maxval, size=tensor.shape).astype(dtype=dtype)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
