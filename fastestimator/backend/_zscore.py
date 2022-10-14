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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def zscore(data: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Apply Zscore processing to a given tensor or array.

    This method can be used with Numpy data:
    ```python
    n = np.array([[0,1],[2,3]])
    b = fe.backend.zscore(n)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0,1],[2,3]])
    b = fe.backend.zscore(t)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0,1],[2,3]])
    b = fe.backend.zscore(p)  # [[-1.34164079, -0.4472136 ],[0.4472136 , 1.34164079]]
    ```

    Args:
        data: The input tensor or array.
        epsilon: A numerical stability constant.

    Returns:
        Data after subtracting mean and divided by standard deviation.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(data):
        data = tf.cast(data, tf.float32)
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        return (data - mean) / tf.maximum(std, epsilon)
    elif isinstance(data, torch.Tensor):
        data = data.type(torch.float32)
        mean = torch.mean(data)
        std = torch.std(data, unbiased=False)
        return (data - mean) / torch.max(std, torch.tensor(epsilon))
    elif isinstance(data, np.ndarray):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / max(std, epsilon)
    else:
        raise ValueError("Unrecognized data type {}".format(type(data)))
