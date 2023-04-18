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
from typing import Collection, TypeVar, Union, overload

import numpy as np
import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)
CollectionT = TypeVar('CollectionT', bound=Collection)


@overload
def to_tensor(data: CollectionT, target_type: str, shared_memory: bool = False) -> CollectionT:
    ...


@overload
def to_tensor(data: Union[Tensor, float, int], target_type: str, shared_memory: bool = False) -> Tensor:
    ...


@overload
def to_tensor(data: None, target_type: str, shared_memory: bool = False) -> None:
    ...


def to_tensor(data: Union[Collection, Tensor, float, int, None], target_type: str,
              shared_memory: bool = False) -> Union[Collection, Tensor, None]:
    """Convert tensors within a collection of `data` to a given `target_type` recursively.

    This method can be used with Numpy data:
    ```python
    data = {"x": np.ones((10,15)), "y":[np.ones((4)), np.ones((5, 3))], "z":{"key":np.ones((2,2))}}
    t = fe.backend.to_tensor(data, target_type='tf')
    # {"x": <tf.Tensor>, "y":[<tf.Tensor>, <tf.Tensor>], "z": {"key": <tf.Tensor>}}
    p = fe.backend.to_tensor(data, target_type='torch')
    # {"x": <torch.Tensor>, "y":[<torch.Tensor>, <torch.Tensor>], "z": {"key": <torch.Tensor>}}
    ```

    This method can be used with TensorFlow tensors:
    ```python
    data = {"x": tf.ones((10,15)), "y":[tf.ones((4)), tf.ones((5, 3))], "z":{"key":tf.ones((2,2))}}
    p = fe.backend.to_tensor(data, target_type='torch')
    # {"x": <torch.Tensor>, "y":[<torch.Tensor>, <torch.Tensor>], "z": {"key": <torch.Tensor>}}
    ```

    This method can be used with PyTorch tensors:
    ```python
    data = {"x": torch.ones((10,15)), "y":[torch.ones((4)), torch.ones((5, 3))], "z":{"key":torch.ones((2,2))}}
    t = fe.backend.to_tensor(data, target_type='tf')
    # {"x": <tf.Tensor>, "y":[<tf.Tensor>, <tf.Tensor>], "z": {"key": <tf.Tensor>}}
    ```

    Args:
        data: A tensor or possibly nested collection of tensors.
        target_type: What kind of tensor(s) to create, one of "tf", "torch", or "np".
        shared_memory: Whether to put the tensor(s) in shared memory (only applicable when `target_type` is 'torch').

    Returns:
        A collection with the same structure as `data`, but with any tensors converted to the `target_type`.
    """
    target_instance = {
        "tf": (tf.Tensor, tf.Variable, tf.distribute.DistributedValues), "torch": torch.Tensor, "np": np.ndarray
    }
    conversion_function = {"tf": tf.convert_to_tensor, "torch": torch.from_numpy, "np": np.array}
    if isinstance(data, target_instance[target_type]):
        if shared_memory and target_type == "torch":
            data.share_memory_()
        return data
    elif data is None:
        return None
    elif isinstance(data, dict):
        return {key: to_tensor(value, target_type) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [to_tensor(val, target_type) for val in data]
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # Named tuple
        return type(data)([to_tensor(val, target_type) for val in data])
    elif isinstance(data, tuple):
        return tuple([to_tensor(val, target_type) for val in data])
    elif isinstance(data, set):
        return set([to_tensor(val, target_type) for val in data])
    else:
        data = conversion_function[target_type](np.array(data))
        if shared_memory and target_type == "torch":
            data.share_memory_()
        return data
