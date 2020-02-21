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


# TODO - Technically the return value here might be a np number if the data is a tf.Tensor
def to_number(data: Union[tf.Tensor, torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    elif isinstance(data, torch.Tensor):
        data = data.data.numpy()
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data.item()
    return data
