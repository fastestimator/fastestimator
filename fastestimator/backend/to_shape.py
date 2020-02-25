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

from typing import Any

import numpy as np


def to_shape(data: Any, add_batch=False, exact_shape=True) -> Any:
    """return the shape of any data in same structure

    Args:
        data: source data
        add_batch: whether to add a batch dimension at the font
        exact_shape: whether to get the exact shape, if False, shape will be filled with None
    Returns:
        shape: data shape with same data structure
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
