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
from typing import TypeVar, Optional

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def squeeze(tensor: Tensor, axis: Optional[int] = None) -> Tensor:
    """Remove an `axis` from a `tensor` if that axis has length 1.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)
    b = fe.backend.squeeze(n)  # [[1, 2], [3, 4], [5, 6]]
    b = fe.backend.squeeze(n, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]
    b = fe.backend.squeeze(n, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)
    b = fe.backend.squeeze(t)  # [[1, 2], [3, 4], [5, 6]]
    b = fe.backend.squeeze(t, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]
    b = fe.backend.squeeze(t, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[[1],[2]]],[[[3],[4]]],[[[5],[6]]]])  # shape == (3, 1, 2, 1)
    b = fe.backend.squeeze(p)  # [[1, 2], [3, 4], [5, 6]]
    b = fe.backend.squeeze(p, axis=1)  # [[[1], [2]], [[3], [4]], [[5], [6]]]
    b = fe.backend.squeeze(p, axis=3)  # [[[1, 2]], [[3, 4]], [[5, 6]]]
    ```

    Args:
        tensor: The input value.
        axis: Which axis to squeeze along, which must have length==1 (or pass None to squeeze all length 1 axes).

    Returns:
        The reshaped `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if isinstance(tensor, tf.Tensor):
        return tf.squeeze(tensor, axis=axis)
    elif isinstance(tensor, torch.Tensor):
        if axis is None:
            return torch.squeeze(tensor)
        else:
            return torch.squeeze(tensor, dim=axis)
    elif isinstance(tensor, np.ndarray):
        return np.squeeze(tensor, axis=axis)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
