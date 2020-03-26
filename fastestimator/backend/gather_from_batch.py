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

from typing import TypeVar

import tensorflow as tf
import torch

from fastestimator.backend.expand_dims import expand_dims

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def gather_from_batch(tensor: Tensor, indices: Tensor) -> Tensor:
    """
    Args:
        tensor: A tensor of shape (batch, ...)
        indices: A tensor of shape (batch, ) or (batch, 1)
    Returns:
        the elements from tensor which were specified by the given indices, shape (batch, ...)
    """
    if len(indices.shape) == 1:  # Indices not batched
        indices = expand_dims(indices, 1)
    if isinstance(tensor, tf.Tensor):
        indices = tf.cast(indices, tf.int64)
        return tf.gather_nd(tensor, indices=indices, batch_dims=1)
    elif isinstance(tensor, torch.Tensor):
        indices = indices.long()
        offset_indices = torch.zeros_like(indices)
        n_indices_per_batch = torch.prod(torch.Tensor(list(tensor.shape)[1:]))
        for i, idx in enumerate(indices):
            offset_indices[i][0] = idx + i * n_indices_per_batch
        return torch.squeeze(torch.take(tensor, offset_indices), dim=-1)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
