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

from typing import TypeVar, Union

import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def clip_by_value(tensor: Tensor, min_value: Union[int, float], max_value: Union[int, float]) -> Tensor:
    if isinstance(tensor, tf.Tensor):
        return tf.clip_by_value(tensor, clip_value_min=min_value, clip_value_max=max_value)
    elif isinstance(tensor, torch.Tensor):
        return tensor.clamp(min=min_value, max=max_value)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
