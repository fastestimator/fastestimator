# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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


def pow(tensor: Tensor, power: Union[int, float, Tensor] ) -> Tensor:
    """Raise a `tensor` to a given `power`.

    This method can be used with Numpy data:
    ```python
    n = np.array([-2, 7, -19])
    b = fe.backend.pow(n, 2)  # [4, 49, 361]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([-2, 7, -19])
    b = fe.backend.pow(t, 2)  # [4, 49, 361]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([-2, 7, -19])
    b = fe.backend.pow(p, 2)  # [4, 49, 361]
    ```

    Args:
        tensor: The input value.
        power: The exponent to raise `tensor` by.

    Returns:
        The exponentiated `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        return tf.pow(tensor, power)
    elif isinstance(tensor, torch.Tensor):
        return tensor.pow(power)
    elif isinstance(tensor, np.ndarray):
        return np.power(tensor, power)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
