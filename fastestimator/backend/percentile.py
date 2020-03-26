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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def percentile(tensor: Tensor,
               percentiles: Union[int, List[int]],
               axis: Union[None, int, List[int]] = None,
               keepdims: bool = True) -> Tensor:
    if isinstance(tensor, tf.Tensor):
        if isinstance(percentiles, List):
            percentiles = tf.convert_to_tensor(percentiles)
        return tfp.stats.percentile(tensor, percentiles, axis=axis, keep_dims=keepdims)
    elif isinstance(tensor, torch.Tensor):
        n_dims = len(tensor.shape)
        if axis is None:
            # Default behavior in tf without axis is to compress all dimensions
            axis = list(range(n_dims))
        # Convert negative axis values to their positive counterparts
        for idx, elem in enumerate(axis):
            axis[idx] = elem % n_dims
        # Extract dims which are not being considered
        other_dims = sorted(set(range(n_dims)).difference(axis))
        # Flatten all of the permutation axis down for kth-value computation
        permutation = other_dims + list(axis)
        permuted = tensor.permute(*permutation)
        other_shape = [tensor.shape[i] for i in other_dims]
        other_shape.append(np.prod([tensor.shape[i] for i in axis]))
        permuted = torch.reshape(permuted, other_shape)
        results = []
        for tile in to_list(percentiles):
            target = min(round(tile / 100.0 * permuted.shape[-1]) + 1, permuted.shape[-1])
            kth_val = torch.kthvalue(permuted, k=target, dim=-1, keepdim=True)[0]
            for dim in range(n_dims - len(kth_val.shape)):
                kth_val = torch.unsqueeze(kth_val, dim=-1)
            # Undo the permutation from earlier
            kth_val = kth_val.permute(*np.argsort(permutation))
            if not keepdims:
                for dim in reversed(axis):
                    kth_val = torch.squeeze(kth_val, dim=dim)
            results.append(kth_val)
        if isinstance(percentiles, int):
            return results[0]
        else:
            return torch.stack(results, dim=0)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
