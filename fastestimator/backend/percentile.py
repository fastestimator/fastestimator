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
import math
from typing import List, TypeVar, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def percentile(tensor: Tensor,
               percentiles: Union[int, List[int]],
               axis: Union[None, int, List[int]] = None,
               keepdims: bool = True) -> Tensor:
    """Compute the `percentiles` of a `tensor`.

    The n-th percentile of `tensor` is the value n/100 of the way from the minimum to the maximum in a sorted copy of
    `tensor`. If the percentile falls in between two values, the lower of the two values will be used.

    This method can be used with Numpy data:
    ```python
    n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = fe.backend.percentile(n, percentiles=[66])  # [[[6]]]
    b = fe.backend.percentile(n, percentiles=[66], axis=0)  # [[[4, 5, 6]]]
    b = fe.backend.percentile(n, percentiles=[66], axis=1)  # [[[2], [5], [8]]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = fe.backend.percentile(t, percentiles=[66])  # [[[6]]]
    b = fe.backend.percentile(t, percentiles=[66], axis=0)  # [[[4, 5, 6]]]
    b = fe.backend.percentile(t, percentiles=[66], axis=1)  # [[[2], [5], [8]]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = fe.backend.percentile(p, percentiles=[66])  # [[[6]]]
    b = fe.backend.percentile(p, percentiles=[66], axis=0)  # [[[4, 5, 6]]]
    b = fe.backend.percentile(p, percentiles=[66], axis=1)  # [[[2], [5], [8]]]
    ```

    Args:
        tensor: The tensor from which to extract percentiles.
        percentiles: One or more percentile values to be computed.
        axis: Along which axes to compute the percentile (None to compute over all axes).
        keepdims: Whether to maintain the number of dimensions from `tensor`.

    Returns:
        The `percentiles` of the given `tensor`.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        if isinstance(percentiles, List):
            percentiles = tf.convert_to_tensor(percentiles)
        return tfp.stats.percentile(tensor, percentiles, axis=axis, keep_dims=keepdims, interpolation='lower')
    elif isinstance(tensor, torch.Tensor):
        n_dims = len(tensor.shape)
        if axis is None:
            # Default behavior in tf without axis is to compress all dimensions
            axis = list(range(n_dims))
        # Convert negative axis values to their positive counterparts
        if isinstance(axis, int):
            axis = [axis]
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
            target = 1 + math.floor(tile / 100.0 * (permuted.shape[-1] - 1))
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
    elif isinstance(tensor, np.ndarray):
        return np.percentile(tensor, percentiles, axis=axis, keepdims=keepdims, interpolation='lower')
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
