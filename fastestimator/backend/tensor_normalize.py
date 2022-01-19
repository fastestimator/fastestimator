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
from typing import Tuple, TypeVar, Union

import torch
import numpy as np
import tensorflow as tf

from fastestimator.backend.to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def normalize(tensor: Tensor, mean: Union[float, Tuple[float, ...]]=(0.485, 0.456, 0.406), std: Union[float, Tuple[float, ...]]=(0.229, 0.224, 0.225), max_pixel_value: float = 255.0, epsilon: float = 1e-7) -> Tensor:
    """Compute the normalized value of a `tensor`.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0)  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0)  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
    b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0)  # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]
    ```

    Args:
        tensor: The input value.
        mean: The mean which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))
        std: The standard deviation which needs to applied(eg: None, 3.8, (1.9, 2.0, 2.9))

    Returns:
        The normalized values of `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    mean = get_mean(tensor, mean, max_pixel_value)
    std = get_std(tensor, std, max_pixel_value)
    epsilon = get_epsilon(tensor, epsilon)

    tensor = (tensor - mean) / (std + epsilon)

    return tensor


def get_mean(tensor: Tensor, mean: Union[float, Tuple[float, ...]]=(0.485, 0.456, 0.406), max_pixel_value: float=255.0) -> Tensor:
    """Get the mean value of a `tensor`.

    Args:
        tensor: The input value.
        mean: The mean which needs to applied(eg: 0.4, (0.485, 0.456, 0.406))
        max_pixel_value: Max value which needs to applied.

    Returns:
        The mean value of `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    mean = np.array(mean, dtype=np.float32)
    mean *= np.array(max_pixel_value)

    if tf.is_tensor(tensor):
        mean = tf.convert_to_tensor(mean)
    elif isinstance(tensor, torch.Tensor):
        mean = to_tensor(mean, "torch")
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return mean


def get_epsilon(tensor: Tensor, epsilon: float) -> Tensor:
    """Convert the epsilon value to right format.

    Args:
        tensor: The input tensor.
        epsilon: The epsilon which needs to added to std(eg: 1e-7)

    Returns:
        The epsilon in right data format with respect to input tensor.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    epsilon = np.array(epsilon, dtype=np.float32)

    if tf.is_tensor(tensor):
        return tf.convert_to_tensor(epsilon)
    elif isinstance(tensor, torch.Tensor):
        return to_tensor(epsilon, "torch")
    elif isinstance(tensor, np.ndarray):
        return epsilon
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))


def get_std(tensor: Tensor, std: Union[float, Tuple[float, ...]]=(0.229, 0.224, 0.225), max_pixel_value: float=255.0) -> Tensor:
    """Get the std value of a `tensor`.

    Args:
        tensor: The input value.
        std: The mean which needs to applied(eg: 0.3, (0.229, 0.224, 0.225))
        max_pixel_value: Max value which needs to applied.

    Returns:
        The std value of `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    std = np.array(std, dtype=np.float32)
    std *= np.array(max_pixel_value)

    if tf.is_tensor(tensor):
        std = tf.convert_to_tensor(std)
    elif isinstance(tensor, torch.Tensor):
        std = to_tensor(std, "torch")
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return std
