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


def check_nan(val: Union[int, float, np.ndarray, tf.Tensor, torch.Tensor]) -> bool:
    """Checks if the input contains NaN values.

    This method can be used with Numpy data:
    ```python
    n = np.array([[[1.0, 2.0], [3.0, np.NaN]], [[5.0, 6.0], [7.0, 8.0]]])
    b = fe.backend.check_nan(n)  # True
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[np.NaN, 6.0], [7.0, 8.0]]])
    b = fe.backend.check_nan(n)  # True
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [np.NaN, 8.0]]])
    b = fe.backend.check_nan(n)  # True
    ```

    Args:
        val: The input value.

    Returns:
        True iff `val` contains NaN
    """
    if tf.is_tensor(val):
        return tf.reduce_any(tf.math.is_nan(val)) or tf.reduce_any(tf.math.is_inf(val))
    elif isinstance(val, torch.Tensor):
        return torch.isnan(val).any() or torch.isinf(val).any()
    else:
        return np.isnan(val).any() or np.isinf(val).any()
