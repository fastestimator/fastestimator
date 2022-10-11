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

from fastestimator.backend._cast import cast

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def where(condition: Tensor, yes: Union[Tensor, int, float], no: Union[Tensor, int, float]) -> Tensor:
    """Compute a tensor based on boolean conditions.

    This method can be used with Numpy data:
    ```python
    n = np.array([[0,1,2],[3,4,5],[6,7,8]])
    b = fe.backend.where(n > 4, n, -1)  # [[-1,-1,-1],[-1,-1,5],[6,7,8]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0,1,2],[3,4,5],[6,7,8]])
    b = fe.backend.where(t > 4, t, -1)  # [[-1,-1,-1],[-1,-1,5],[6,7,8]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
    b = fe.backend.where(p > 4, p, -1)  # [[-1,-1,-1],[-1,-1,5],[6,7,8]]
    ```

    Args:
        condition: A tensor of boolean conditions
        yes: The value to insert if the condition is True
        no: The value to insert if the condition is False

    Returns:
        A tensor composed of `yes` and `no` values according to the `conditions`

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(condition):
        return tf.where(condition, yes, no)
    elif isinstance(condition, torch.Tensor):
        if isinstance(yes, torch.Tensor):
            no = cast(no, yes)
        elif isinstance(no, torch.Tensor):
            yes = cast(yes, no)
        return torch.where(condition, yes, no)
    elif isinstance(condition, np.ndarray):
        return np.where(condition, yes, no)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(condition)))
