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
from typing import List, Optional, TypeVar

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def concat(tensors: List[Tensor], axis: int = 0) -> Optional[Tensor]:
    """Concatenate a list of `tensors` along a given `axis`.

    This method can be used with Numpy data:
    ```python
    n = [np.array([[0, 1]]), np.array([[2, 3]]), np.array([[4, 5]])]
    b = fe.backend.concat(n, axis=0)  # [[0, 1], [2, 3], [4, 5]]
    b = fe.backend.concat(n, axis=1)  # [[0, 1, 2, 3, 4, 5]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = [tf.constant([[0, 1]]), tf.constant([[2, 3]]), tf.constant([[4, 5]])]
    b = fe.backend.concat(t, axis=0)  # [[0, 1], [2, 3], [4, 5]]
    b = fe.backend.concat(t, axis=1)  # [[0, 1, 2, 3, 4, 5]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])]
    b = fe.backend.concat(p, axis=0)  # [[0, 1], [2, 3], [4, 5]]
    b = fe.backend.concat(p, axis=1)  # [[0, 1, 2, 3, 4, 5]]
    ```

    Args:
        tensors: A list of tensors to be concatenated.
        axis: The axis along which to concatenate the input.

    Returns:
        A concatenated representation of the `tensors`, or None if the list of `tensors` was empty.

    Raises:
        ValueError: If `tensors` is an unacceptable data type.
    """
    if len(tensors) == 0:
        return None
    if tf.is_tensor(tensors[0]):
        return tf.concat(tensors, axis=axis)
    elif isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors, dim=axis)
    elif isinstance(tensors[0], np.ndarray):
        return np.concatenate(tensors, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensors[0])))
