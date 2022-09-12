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
from typing import Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as T

from fastestimator.backend._convert_tensor_precision import convert_tensor_precision

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def normalize(tensor: Tensor,
              mean: Union[float, Sequence[float]] = (0.485, 0.456, 0.406),
              std: Union[float, Sequence[float]] = (0.229, 0.224, 0.225),
              max_pixel_value: float = 255.0,
              epsilon: float = 1e-7) -> Tensor:
    """
        Compute the normalized value of a `tensor`.

        This method can be used with Numpy data:
        python
        n = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
        b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0) # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]


        This method can be used with TensorFlow tensors:
        python
        t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
        b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0) # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]


        This method can be used with PyTorch tensors:
        python
        p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        b = fe.backend.tensor_normalize(n, 0.5625, 0.2864, 8.0)  # ([[[-1.52752516, -1.0910894 ], [-0.65465364, -0.21821788]], [[ 0.21821788,  0.65465364], [ 1.0910894 ,  1.52752516]]])
        b = fe.backend.tensor_normalize(n, (0.5, 0.625), (0.2795, 0.2795), 8.0) # [[[-1.34164073, -1.34164073], [-0.44721358, -0.44721358]], [[ 0.44721358,  0.44721358], [ 1.34164073,  1.34164073]]]

        Args:
            tensor: The input 'tensor' value.
            mean: The mean which needs to applied(eg: 3.8, (0.485, 0.456, 0.406)).
            std: The standard deviation which needs to applied(eg: 3.8, (0.229, 0.224, 0.225)).
            max_pixel_value: The max value of the input data(eg: 255, 65025) to be multipled with mean and std to get actual mean and std.
                            To directly use the mean and std provide set max_pixel_value as 1.
            epsilon: Default value to be added to std to avoid divide by zero error.

        Returns:
            The normalized values of `tensor`.

        Raises:
            ValueError: If `tensor` is an unacceptable data type.
    """
    framework, device = get_framework(tensor)

    mean = convert_tensor_precision(get_scaled_data(mean, max_pixel_value, framework, device))
    std = convert_tensor_precision(get_scaled_data(std, max_pixel_value, framework, device))
    tensor = convert_tensor_precision(tensor)

    if framework == 'torch':
        tensor = T.Normalize(mean=mean, std=std + epsilon)(tensor)
    else:
        tensor = (tensor - mean) / (std + epsilon)

    return tensor


def get_framework(tensor: Tensor) -> Tuple[str, Optional[str]]:
    """
        Get the framework of the input data.

        Args:
            tensor: The input tensor.
        Returns:
            framework: Framework which is used to load input data.
            device: The device on which the method is executed (Eg. cuda, cpu). Only applicable to torch.
    """
    device = None
    if tf.is_tensor(tensor):
        framework = 'tf'
    elif isinstance(tensor, torch.Tensor):
        framework = 'torch'
        device = tensor.device
    elif isinstance(tensor, np.ndarray):
        framework = 'np'
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))

    return (framework, device)


def get_scaled_data(data: Union[float, Sequence[float]] = (0.485, 0.456, 0.406),
                    scale_factor: float = 255.0,
                    framework: str = 'np',
                    device: Optional[torch.device] = None) -> Tensor:
    """
        Get the scaled value of a input data.

        Args:
            data: The data which needs to be scaled. (eg: 0.4, (0.485, 0.456, 0.406)).
            scale_factor: Scale factor which needs to be multipled with input data.
            framework: Framework currently method is running in.(Eg: 'np','tf', 'torch').
            device: Current device. (eg: 'cpu','cuda').

        Returns:
            The scaled value of input data.
    """
    if framework == 'torch':
        data = torch.tensor(data, device=device)
    elif framework == 'tf':
        data = tf.constant(data)
    else:
        data = np.array(data)

    return data * scale_factor
