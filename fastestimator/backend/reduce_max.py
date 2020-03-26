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

from typing import TypeVar, Union, List

import tensorflow as tf
import torch

from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def reduce_max(tensor: Tensor, axis: Union[None, int, List[int]] = None, keepdims: bool = False) -> Tensor:
    if isinstance(tensor, tf.Tensor):
        return tf.reduce_max(tensor, axis=axis, keepdims=keepdims)
    elif isinstance(tensor, torch.Tensor):
        if axis is None:
            axis = list(range(len(tensor.shape)))
        axis = to_list(axis)
        axis = reversed(sorted(axis))
        for ax in axis:
            tensor = tensor.max(dim=ax, keepdim=keepdims)[0]
        return tensor
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
