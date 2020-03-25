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

from fastestimator.util.util import STRING_TO_TORCH_DTYPE

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def zeros_like(tensor: Tensor, dtype: Union[None, str] = None) -> Tensor:
    if isinstance(tensor, tf.Tensor):
        return tf.zeros_like(tensor, dtype=dtype)
    elif isinstance(tensor, torch.Tensor):
        return torch.zeros_like(tensor, dtype=STRING_TO_TORCH_DTYPE[dtype])
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
