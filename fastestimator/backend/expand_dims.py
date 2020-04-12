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


def expand_dims(tensor: Tensor, axis: int = 1) -> Tensor:
    """Create a new dimension in `tensor` along a given `axis`.

    This method can be used with Numpy data:
    ```python
    n = np.array([2,7,5])
    b = fe.backend.expand_dims(n, axis=0)  # [[2, 5, 7]]
    b = fe.backend.expand_dims(n, axis=1)  # [[2], [5], [7]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([2,7,5])
    b = fe.backend.expand_dims(t, axis=0)  # [[2, 5, 7]]
    b = fe.backend.expand_dims(t, axis=1)  # [[2], [5], [7]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([2,7,5])
    b = fe.backend.expand_dims(p, axis=0)  # [[2, 5, 7]]
    b = fe.backend.expand_dims(p, axis=1)  # [[2], [5], [7]]
    ```

    Args:
        tensor: The input to be modified, having n dimensions.
        axis: Which axis should the new axis be inserted along. Must be in the range [-n-1, n].

    Returns:
        A concatenated representation of the `tensors`, or None if the list of `tensors` was empty.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.expand_dims(tensor, axis=axis)
    elif isinstance(tensor, torch.Tensor):
        return torch.unsqueeze(tensor, dim=axis)
    elif isinstance(tensor, np.ndarray):
        return np.expand_dims(tensor, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
