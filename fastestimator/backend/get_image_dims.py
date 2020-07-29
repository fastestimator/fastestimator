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


def get_image_dims(tensor: Tensor) -> Tensor:
    """Get the `tensor` height, width and channels.

    This method can be used with Numpy data:
    ```python
    n = np.random.random((2, 12, 12, 3))
    b = fe.backend.get_image_dims(n)  # (3, 12, 12)
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.random.uniform((2, 12, 12, 3))
    b = fe.backend.get_image_dims(t)  # (3, 12, 12)
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.rand((2, 3, 12, 12))
    b = fe.backend.get_image_dims(p)  # (3, 12, 12)
    ```

    Args:
        tensor: The input tensor.

    Returns:
        Channels, height and width of the `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    assert len(tensor.shape) == 3 or len(tensor.shape) == 4, "Number of dimensions of input must be either 3 or 4"
    shape_length = len(tensor.shape)
    if tf.is_tensor(tensor) or isinstance(tensor, np.ndarray):
        return tensor.shape[-1], tensor.shape[-3], tensor.shape[-2]
    elif isinstance(tensor, torch.Tensor):
        return tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
