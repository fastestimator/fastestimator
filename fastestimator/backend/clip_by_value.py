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

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def clip_by_value(tensor: Tensor, min_value: Union[int, float, Tensor], max_value: Union[int, float, Tensor]) -> Tensor:
    """Clip a tensor such that `min_value` <= tensor <= `max_value`.

    This method can be used with Numpy data:
    ```python
    n = np.array([-5, 4, 2, 0, 9, -2])
    b = fe.backend.clip_by_value(n, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([-5, 4, 2, 0, 9, -2])
    b = fe.backend.clip_by_value(t, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([-5, 4, 2, 0, 9, -2])
    b = fe.backend.clip_by_value(p, min_value=-2, max_value=3)  # [-2, 3, 2, 0, 3, -2]
    ```

    Args:
        tensor: The input value.
        min_value: The minimum value to clip to.
        max_value: The maximum value to clip to.

    Returns:
        The `tensor`, with it's values clipped.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.clip_by_value(tensor, clip_value_min=min_value, clip_value_max=max_value)
    elif isinstance(tensor, torch.Tensor):
        if isinstance(min_value, torch.Tensor):
            min_value = min_value.item()
        if isinstance(max_value, torch.Tensor):
            max_value = max_value.item()
        return tensor.clamp(min=min_value, max=max_value)
    elif isinstance(tensor, np.ndarray):
        return np.clip(tensor, a_min=min_value, a_max=max_value)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
