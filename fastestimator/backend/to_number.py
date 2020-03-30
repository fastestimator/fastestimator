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


def to_number(data: Union[tf.Tensor, torch.Tensor, np.ndarray]) -> np.ndarray:
    """convert tensor values to python values

    Args:
        data: any tensor value

    Returns:
        np.ndarray: python value of the tensor
    """
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach().numpy()
        else:
            data = data.numpy()
    return np.array(data)
