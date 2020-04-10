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
from typing import Union

import numpy as np
import tensorflow as tf
import torch


def to_number(data: Union[tf.Tensor, torch.Tensor, np.ndarray, int, float]) -> np.ndarray:
    """Convert an input value into a Numpy ndarray.

    This method can be used with Python and Numpy data:
    ```python
    b = fe.backend.to_number(5)  # 5 (type==np.ndarray)
    b = fe.backend.to_number(4.0)  # 4.0 (type==np.ndarray)
    n = np.array([1, 2, 3])
    b = fe.backend.to_number(n)  # [1, 2, 3] (type==np.ndarray)
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([1, 2, 3])
    b = fe.backend.to_number(t)  # [1, 2, 3] (type==np.ndarray)
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([1, 2, 3])
    b = fe.backend.to_number(p)  # [1, 2, 3] (type==np.ndarray)
    ```

    Args:
        data: The value to be converted into a np.ndarray.

    Returns:
        An ndarray corresponding to the given `data`.
    """
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach().numpy()
        else:
            data = data.numpy()
    return np.array(data)
