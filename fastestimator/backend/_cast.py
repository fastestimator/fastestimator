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

from fastestimator.util.util import STRING_TO_TF_DTYPE, STRING_TO_TORCH_DTYPE

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def cast(data: Union[Collection, Tensor], dtype: Union[str, Tensor]) -> Union[Collection, Tensor]:
    """Cast the data to a specific data type recursively.

   This method can be used with Numpy data:
    ```python
    data = {"x": np.ones((10,15)), "y":[np.ones((4)), np.ones((5, 3))], "z":{"key":np.ones((2,2))}}
    fe.backend.to_type(data)
    # {'x': dtype('float64'), 'y': [dtype('float64'), dtype('float64')], 'z': {'key': dtype('float64')}}
    data = fe.backend.cast(data, "float16")
    fe.backend.to_type(data)
    # {'x': dtype('float16'), 'y': [dtype('float16'), dtype('float16')], 'z': {'key': dtype('float16')}}
    ```

    This method can be used with TensorFlow tensors:
    ```python
    data = {"x": tf.ones((10,15)), "y":[tf.ones((4)), tf.ones((5, 3))], "z":{"key":tf.ones((2,2))}}
    fe.backend.to_type(data) # {'x': tf.float32, 'y': [tf.float32, tf.float32], 'z': {'key': tf.float32}}
    data = fe.backend.cast(data, "uint8")
    fe.backend.to_type(data) # {'x': tf.uint8, 'y': [tf.uint8, tf.uint8], 'z': {'key': tf.uint8}}
    ```

    This method can be used with PyTorch tensors:
    ```python
    data = {"x": torch.ones((10,15)), "y":[torch.ones((4)), torch.ones((5, 3))], "z":{"key":torch.ones((2,2))}}
    fe.backend.to_type(data) # {'x': torch.float32, 'y': [torch.float32, torch.float32], 'z': {'key': torch.float32}}
    data = fe.backend.cast(data, "float64")
    fe.backend.to_type(data) # {'x': torch.float64, 'y': [torch.float64, torch.float64], 'z': {'key': torch.float64}}
    ```

    Args:
        data: A tensor or possibly nested collection of tensors.
        dtype: Target reference data type, can be one of following: uint8, int8, int16, int32, int64, float16, float32, float64. Tensor.

    Returns:
        A collection with the same structure as `data` with reference data type.
    """
    if isinstance(dtype, str):
        if isinstance(data, dict):
            return {key: cast(value, dtype) for (key, value) in data.items()}
        elif isinstance(data, list):
            return [cast(val, dtype) for val in data]
        elif isinstance(data, tuple):
            return tuple([cast(val, dtype) for val in data])
        elif isinstance(data, set):
            return set([cast(val, dtype) for val in data])
        elif tf.is_tensor(data):
            return tf.cast(data, STRING_TO_TF_DTYPE[dtype])
        elif isinstance(data, torch.Tensor):
            return data.type(STRING_TO_TORCH_DTYPE[dtype])
        else:
            return np.array(data, dtype=dtype)
    elif tf.is_tensor(dtype) or isinstance(dtype, torch.Tensor) or isinstance(dtype, np.ndarray):
        if tf.is_tensor(dtype):
            return tf.cast(data, dtype.dtype)
        elif isinstance(dtype, torch.Tensor):
            if isinstance(data, torch.Tensor):
                return data.to(dtype.dtype)
            return torch.tensor(data, dtype=dtype.dtype, device=dtype.device)
        else:
            return np.array(data, dtype=dtype.dtype)
    else:
        ValueError("Unexpected reference data type.")
