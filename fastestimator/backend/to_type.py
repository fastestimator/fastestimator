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
from typing import Collection, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def to_type(data: Union[Collection, Tensor]) -> Union[Collection, str]:
    """Compute the data types of tensors within a collection of `data`recursively.

    This method can be used with Numpy data:
    ```python
    data = {"x": np.ones((10,15), dtype="float32"), "y":[np.ones((4), dtype="int8"), np.ones((5, 3), dtype="double")],
        "z":{"key":np.ones((2,2), dtype="int64")}}
    types = fe.backend.to_type(data)
    # {'x': dtype('float32'), 'y': [dtype('int8'), dtype('float64')], 'z': {'key': dtype('int64')}}
    ```

    This method can be used with TensorFlow tensors:
    ```python
    data = {"x": tf.ones((10,15), dtype="float32"), "y":[tf.ones((4), dtype="int8"), tf.ones((5, 3), dtype="double")],
        "z":{"key":tf.ones((2,2), dtype="int64")}}
    types = fe.backend.to_type(data)
    # {'x': tf.float32, 'y': [tf.int8, tf.float64], 'z': {'key': tf.int64}}
    ```

    This method can be used with PyTorch tensors:
    ```python
    data = {"x": torch.ones((10,15), dtype=torch.float32), "y":[torch.ones((4), dtype=torch.int8), torch.ones((5, 3),
        dtype=torch.double)], "z":{"key":torch.ones((2,2), dtype=torch.long)}}
    types = fe.backend.to_type(data)
    # {'x': torch.float32, 'y': [torch.int8, torch.float64], 'z': {'key': torch.int64}}
    ```

    Args:
        data: A tensor or possibly nested collection of tensors.

    Returns:
        A collection with the same structure as `data`, but with any tensors substituted for their dtypes.
    """
    if isinstance(data, dict):
        return {key: to_type(value) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [to_type(val) for val in data]
    elif isinstance(data, tuple):
        return tuple([to_type(val) for val in data])
    elif isinstance(data, set):
        return set([to_type(val) for val in data])
    elif hasattr(data, "dtype"):
        return data.dtype
    else:
        return np.array(data).dtype
