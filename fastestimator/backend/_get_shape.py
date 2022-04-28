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
from typing import TypeVar

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def get_shape(tensor: Tensor) -> Tensor:
    """Find shape of a given `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.get_shape(n)  # [3,2,2]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.get_shape(t)  # [3,2,2]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    b = fe.backend.get_shape(p)  # [3,2,2]
    ```

    Args:
        tensor: The tensor to find shape of.

    Returns:
        Shape of the given 'tensor'.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.shape(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor.shape
    elif isinstance(tensor, np.ndarray):
        return tensor.shape
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
