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


def to_shape(data: Union[Collection, Tensor], add_batch=False, exact_shape=True) -> Union[Collection, Tensor]:
    """Compute the shape of tensors within a collection of `data`recursively.

    This method can be used with Numpy data:
    ```python
    data = {"x": np.ones((10,15)), "y":[np.ones((4)), np.ones((5, 3))], "z":{"key":np.ones((2,2))}}
    shape = fe.backend.to_shape(data)  # {"x": (10, 15), "y":[(4), (5, 3)], "z": {"key": (2, 2)}}
    shape = fe.backend.to_shape(data, add_batch=True)
    # {"x": (None, 10, 15), "y":[(None, 4), (None, 5, 3)], "z": {"key": (None, 2, 2)}}
    shape = fe.backend.to_shape(data, exact_shape=False)
    # {"x": (None, None), "y":[(None), (None, None)], "z": {"key": (None, None)}}
    ```

    This method can be used with TensorFlow tensors:
    ```python
    data = {"x": tf.ones((10,15)), "y":[tf.ones((4)), tf.ones((5, 3))], "z":{"key":tf.ones((2,2))}}
    shape = fe.backend.to_shape(data)  # {"x": (10, 15), "y":[(4), (5, 3)], "z": {"key": (2, 2)}}
    shape = fe.backend.to_shape(data, add_batch=True)
    # {"x": (None, 10, 15), "y":[(None, 4), (None, 5, 3)], "z": {"key": (None, 2, 2)}}
    shape = fe.backend.to_shape(data, exact_shape=False)
    # {"x": (None, None), "y":[(None), (None, None)], "z": {"key": (None, None)}}
    ```

    This method can be used with PyTorch tensors:
    ```python
    data = {"x": torch.ones((10,15)), "y":[torch.ones((4)), torch.ones((5, 3))], "z":{"key":torch.ones((2,2))}}
    shape = fe.backend.to_shape(data)  # {"x": (10, 15), "y":[(4), (5, 3)], "z": {"key": (2, 2)}}
    shape = fe.backend.to_shape(data, add_batch=True)
    # {"x": (None, 10, 15), "y":[(None, 4), (None, 5, 3)], "z": {"key": (None, 2, 2)}}
    shape = fe.backend.to_shape(data, exact_shape=False)
    # {"x": (None, None), "y":[(None), (None, None)], "z": {"key": (None, None)}}
    ```

    Args:
        data: A tensor or possibly nested collection of tensors.
        add_batch: Whether to prepend a batch dimension to the shapes.
        exact_shape: Whether to return the exact shapes, or if False to fill the shapes with None values.

    Returns:
        A collection with the same structure as `data`, but with any tensors substituted for their shapes.
    """
    if isinstance(data, dict):
        return {key: to_shape(value, add_batch, exact_shape) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [to_shape(val, add_batch, exact_shape) for val in data]
    elif isinstance(data, tuple):
        return tuple([to_shape(val, add_batch, exact_shape) for val in data])
    elif isinstance(data, set):
        return set([to_shape(val, add_batch, exact_shape) for val in data])
    elif hasattr(data, "shape"):
        shape = data.shape
        if not exact_shape:
            shape = [None] * len(shape)
        if add_batch:
            shape = [None] + list(shape)
        return shape
    else:
        return to_shape(np.array(data), add_batch, exact_shape)
