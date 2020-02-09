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
import tensorflow as tf
import torch


def to_tensor(data: Any, target_type: str) -> Any:
    """convert the value of any data sturcture to tensor

    Args:
        data (Any): source data
        target_type (str): target data type, can be either "tensorflow" or "pytorch"

    Returns:
        data: target data
    """
    target_instance = {"tensorflow": tf.Tensor, "torch": torch.Tensor}
    conversion_function = {"tensorflow": tf.convert_to_tensor, "torch": torch.from_numpy}
    if isinstance(data, target_instance[target_type]):
        return data
    elif isinstance(data, dict):
        return {key: to_tensor(value, target_type) for (key, value) in data.items()}
    elif isinstance(data, list):
        return [to_tensor(val, target_type) for val in data]
    elif isinstance(data, tuple):
        return tuple([to_tensor(val, target_type) for val in data])
    elif isinstance(data, set):
        return set([to_tensor(val, target_type) for val in data])
    else:
        return conversion_function[target_type](np.array(data))
