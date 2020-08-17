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
from typing import Optional, TypeVar

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.maximum import maximum
from fastestimator.backend.reduce_sum import reduce_sum
from fastestimator.backend.reshape import reshape
from fastestimator.backend.tensor_pow import tensor_pow
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.util.util import TENSOR_TO_NP_DTYPE

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


def iwd(tensor: Tensor,
        power: float = 1.0,
        max_prob: float = 0.95,
        pairwise_distance: float = 1.0,
        eps: Optional[Tensor] = None) -> Tensor:
    """Compute the Inverse Weighted Distance from the given input.

    This can be used as an activation function for the final layer of a neural network instead of softmax. For example,
    instead of: model.add(layers.Dense(classes, activation='softmax')), you could use:
    model.add(layers.Dense(classes, activation=lambda x: iwd(tf.nn.sigmoid(x))))

    This method can be used with Numpy data:
    ```python
    n = np.array([[0.5]*5, [0]+[1]*4])
    b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
    ```

    This method can be used with TensorFlow tensors:
    ```python
    t = tf.constant([[0.5]*5, [0]+[1]*4])
    b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
    ```

    This method can be used with PyTorch tensors:
    ```python
    p = torch.tensor([[0.5]*5, [0]+[1]*4])
    b = fe.backend.iwd(n)  # [[0.2, 0.2, 0.2, 0.2, 0.2], [0.95, 0.0125, 0.0125, 0.0125, 0.0125]]
    ```

    Args:
        tensor: The input value. Should be of shape (Batch, C) where every element in C corresponds to a (non-negative)
            distance to a target class.
        power: The power to raise the inverse distances to. 1.0 results in a fairly intuitive probability output. Larger
            powers can widen regions of certainty, whereas values between 0 and 1 can widen regions of uncertainty.
        max_prob: The maximum probability to assign to a class estimate when it is distance zero away from the target.
            For numerical stability this must be less than 1.0. We have found that using smaller values like 0.95 can
            lead to natural adversarial robustness.
        pairwise_distance: The distance to any other class when the distance to a target class is zero. For example, if
            you have a perfect match for class 'a', what distance should be reported to class 'b'. If you have a metric
            where this isn't constant, just use an approximate expected distance. In that case `max_prob` will only give
            you approximate control over the true maximum probability.
        eps: The numeric stability constant to be used when d approaches zero. If None then it will be computed using
            `max_prob` and `pairwise_distance`. If not None, then `max_prob` and `pairwise_distance` will be ignored.

    Returns:
        A probability distribution of shape (Batch, C) where smaller distances from `tensor` correspond to larger
        probabilities.
    """
    if eps is None:
        eps = np.array(pairwise_distance * math.pow((1.0 - max_prob) / (max_prob * (tensor.shape[-1] - 1)), 1 / power),
                       dtype=TENSOR_TO_NP_DTYPE[tensor.dtype])
        eps = to_tensor(
            eps, target_type='torch' if isinstance(tensor, torch.Tensor) else 'tf' if tf.is_tensor(tensor) else 'np')
        if isinstance(eps, torch.Tensor):
            eps = eps.to("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = maximum(tensor, eps)
    tensor = tensor_pow(1.0 / tensor, power)
    tensor = tensor / reshape(reduce_sum(tensor, axis=-1), shape=[-1, 1])
    return tensor
