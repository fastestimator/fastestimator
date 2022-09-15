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
from typing import Sequence, TypeVar

import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def resize_3d(tensor: Tensor, output_shape: Sequence[int], resize_mode: str = 'nearest') -> Tensor:
    """Reshape a `tensor` to conform to a given shape.Currently torch doesn't support 16 bit tensors on cpu.

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[[[0.], [1.]], [[2.], [3.]]], [[[4.], [5.]], [[6.], [7.]]]]])
    b = fe.backend.resize_3d(t, output_shape=[3, 3, 3])  # [[[[[0.], [0.], [1.], [1.]], [[0.], [0.], [1.], [1.]], [[2.], [2.], [3.], [3.]], [[2.], [2.], [3.], [3.]]],
                                                            [[[0.], [0.], [1.], [1.]], [[0.], [0.], [1.], [1.]], [[2.], [2.], [3.], [3.]], [[2.], [2.], [3.], [3.]]],
                                                            [[[4.], [4.], [5.], [5.]], [[4.], [4.], [5.], [5.]], [[6.], [6.], [7.], [7.]], [[6.], [6.], [7.], [7.]]],
                                                            [[[4.], [4.], [5.], [5.]], [[4.], [4.], [5.], [5.]], [[6.], [6.], [7.], [7.]], [[6.], [6.], [7.], [7.]]]]]
    ```

    This method can be used with PyTorch tensors:

    ```python
    p = torch.tensor([[[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]]])
    b = fe.backend.resize_3d(p, output_shape=[3, 3, 3])  # [[[[[0., 0., 1., 1.], [0., 0., 1., 1.], [2., 2., 3., 3.], [2., 2., 3., 3.]],
                                                              [[0., 0., 1., 1.], [0., 0., 1., 1.], [2., 2., 3., 3.], [2., 2., 3., 3.]],
                                                              [[4., 4., 5., 5.], [4., 4., 5., 5.], [6., 6., 7., 7.], [6., 6., 7., 7.]],
                                                              [[4., 4., 5., 5.], [4., 4., 5., 5.], [6., 6., 7., 7.], [6., 6., 7., 7.]]]]]
    ```

    Args:
        tensor: The input value.
        output_shape: The new size of the tensor.
        resize_mode: mode to apply for resizing

    Returns:
        The resized `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    assert resize_mode in ['nearest', 'area'], "Only following resize modes are supported: 'nearest', 'area' "

    if tf.is_tensor(tensor):
        return resize_tensorflow_tensor(tensor, output_shape, resize_mode)
    elif isinstance(tensor, torch.Tensor):
        return torch.nn.functional.interpolate(tensor, output_shape, mode=resize_mode)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def resize_tensorflow_tensor(data: tf.Tensor, output_shape: Sequence[int], resize_mode: str) -> tf.Tensor:
    """
        Resize tensorflow tensor

        Input:
            data: Input tensorflow tensor
            output_shape: (X, Y, Z) Expected output shape of tensor
    """
    d1_new, d2_new, d3_new = output_shape
    data_shape = tf.shape(data)
    batch_size, d1, d2, d3, c = data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4]

    # resize d2-d3
    squeeze_b_x = tf.reshape(data, [-1, d2, d3, c])
    resize_b_x = tf.image.resize(squeeze_b_x, [d2_new, d3_new], resize_mode)
    resume_b_x = tf.reshape(resize_b_x, [batch_size, d1, d2_new, d3_new, c])

    # resize d1
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    squeeze_b_z = tf.reshape(reoriented, [-1, d2_new, d1, c])
    resize_b_z = tf.image.resize(squeeze_b_z, [d2_new, d1_new], resize_mode)
    resume_b_z = tf.reshape(resize_b_z, [batch_size, d3_new, d2_new, d1_new, c])
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor
