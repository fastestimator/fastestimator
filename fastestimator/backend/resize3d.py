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

import tensorflow as tf
import torch
import torchvision

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def resize_3d(tensor: Tensor, size: List[int]) -> Tensor:
    """Reshape a `tensor` to conform to a given shape.

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[[[0.], [1.]], [[2.], [3.]]], [[[4.], [5.]], [[6.], [7.]]]]])
    b = fe.backend.resize_3d(t, shape=[3, 3, 3])  # [[[[[0.], [0.5], [1.]], [[1.], [1.5], [2.]], [[2.], [2.5], [3.]]],
                                                      [[[2.], [2.5], [3.]], [[3.], [3.5], [4.]], [[4.], [4.5], [5.]]], [6.]], [[6.], [6.5], [7.]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]]])
    b = fe.backend.resize_3d(p, shape=[3, 3, 3])  # [[[[[0., 0.5, 1.], [1., 1.5, 2.], [2., 2.5, 3.]],
                                                       [[2., 2.5, 3.], [3., 3.5, 4.], [4., 4.5, 5.]],
                                                       [[4., 4.5, 5.], [5., 5.5, 6.], [6., 6.499999, 7.]]]]]
    ```

    Args:
        tensor: The input value.
        size: The new size of the tensor.

    Returns:
        The resized `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return resize_tensorflow_tensor(tensor, size)
    elif isinstance(tensor, torch.Tensor):
        return resize_pytorch_tensor(tensor, size)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def resize_tensorflow_tensor(data: tf.Tensor, size: List[int]) -> tf.Tensor:

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


def resize_pytorch_tensor(pytorch_array: torch.Tensor, size: List[int]) -> torch.Tensor:

    d1_new, d2_new, d3_new = size

    data_shape = pytorch_array.shape
    batch_size, c, d1, d2, d3 = data_shape[0], data_shape[1], data_shape[2], data_shape[3], data_shape[4]

    # resize d2-d3
    permute_pytorch_array = pytorch_array.permute((0, 2, 1, 3, 4))
    squeeze_b_x = torch.reshape(permute_pytorch_array, [-1, c, d2, d3])
    resize_b_x = torchvision.transforms.functional.resize(squeeze_b_x, [d2_new, d3_new])
    resume_b_x = torch.reshape(resize_b_x, [batch_size, d1, c, d2_new, d3_new])

    # resize d1
    reoriented = resume_b_x.permute((0, 4, 2, 3, 1))
    squeeze_b_z = torch.reshape(reoriented, [-1, c, d2_new, d1])
    resize_b_z = torchvision.transforms.functional.resize(squeeze_b_z, [d2_new, d1_new])
    resume_b_z = torch.reshape(resize_b_z, [batch_size, d3_new, c, d2_new, d1_new])
    output_tensor = resume_b_z.permute((0, 2, 4, 3, 1))
    return output_tensor
