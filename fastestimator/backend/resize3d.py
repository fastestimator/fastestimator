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
from typing import List, TypeVar

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def resize_3d(tensor: Tensor, shape: List[int]) -> Tensor:
    """Reshape a `tensor` to conform to a given shape.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(n, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(n, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(n, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(n, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(t, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(t, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(t, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(t, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.reshape(p, shape=[-1])  # [1, 2, 3, 4, 5, 6, 7, 8]
    b = fe.backend.reshape(p, shape=[2, 4])  # [[1, 2, 3, 4], [5, 6, 7, 8]]
    b = fe.backend.reshape(p, shape=[4, 2])  # [[1, 2], [3, 4], [5, 6], [7, 8]]
    b = fe.backend.reshape(p, shape=[2, 2, 2, 1])  # [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]
    ```

    Args:
        tensor: The input value.
        shape: The new shape of the tensor. At most one value may be -1 which indicates that whatever values are left
            should be packed into that axis.

    Returns:
        The reshaped `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return resize_tensorflow_tensor(tensor, shape)
    elif isinstance(tensor, torch.Tensor):
        return torch.nn.functional.interpolate(tensor, shape)
    elif isinstance(tensor, np.ndarray):
        return np.reshape(tensor, shape)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def resize_tensorflow_tensor(data, size):
    d1_new, d2_new, d3_new = size
    data_shape = tf.shape(data)
    batch_size, d1, d2, d3, c = data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4]
    # resize d2-d3
    squeeze_b_x = tf.reshape(data, [-1, d2, d3, c])
    resize_b_x = tf.image.resize(squeeze_b_x, [d2_new, d3_new])
    resume_b_x = tf.reshape(resize_b_x, [batch_size, d1, d2_new, d3_new, c])
    # resize d1
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    squeeze_b_z = tf.reshape(reoriented, [-1, d2_new, d1, c])
    resize_b_z = tf.image.resize(squeeze_b_z, [d2_new, d1_new])
    resume_b_z = tf.reshape(resize_b_z, [batch_size, d3_new, d2_new, d1_new, c])
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor