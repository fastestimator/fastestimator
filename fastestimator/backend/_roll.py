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
from typing import List, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def roll(tensor: Tensor, shift: Union[int, List[int]], axis: Union[int, List[int]]) -> Tensor:
    """Roll a `tensor` elements along a given axis.

    The elements are shifted forward or reverse direction by the offset of `shift`. Overflown elements beyond the last
    position will be re-introduced at the first position.

    This method can be used with Numpy data:
    ```python
    n = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    b = fe.backend.roll(n, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
    b = fe.backend.roll(n, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
    b = fe.backend.roll(n, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
    b = fe.backend.roll(n, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    b = fe.backend.roll(t, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
    b = fe.backend.roll(t, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
    b = fe.backend.roll(t, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
    b = fe.backend.roll(t, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    b = fe.backend.roll(p, shift=1, axis=0)  # [[5, 6, 7], [1, 2, 3]]
    b = fe.backend.roll(p, shift=2, axis=1)  # [[2, 3, 1], [6, 7, 5]]
    b = fe.backend.roll(p, shift=-2, axis=1)  # [[3, 1, 2], [7, 5, 6]]
    b = fe.backend.roll(p, shift=[-1, -1], axis=[0, 1])  # [[6, 7, 5], [2, 3, 1]]
    ```

    Args:
        tensor: The input value.
        shift: The number of places by which the elements need to be shifted. If shift is a list, axis must be a list of
            same size.
        axis: axis along which elements will be rolled.

    Returns:
        The rolled `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.roll(tensor, shift=shift, axis=axis)
    elif isinstance(tensor, torch.Tensor):
        return torch.roll(tensor, shifts=shift, dims=axis)
    elif isinstance(tensor, np.ndarray):
        return np.roll(tensor, shift=shift, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
