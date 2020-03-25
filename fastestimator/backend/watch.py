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

from typing import TypeVar, Optional

import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, torch.autograd.Variable)


def watch(tensor: Tensor, tape: Optional[tf.GradientTape] = None) -> Tensor:
    """ Monitor a given tensor for gradient computations
    Args:
        tensor: The tensor to be monitored
        tape: A gradient tape to track gradients on (applicable for tf backend)
    Returns:
        The tensor to be monitored
    """
    if isinstance(tensor, tf.Tensor):
        tape.watch(tensor)
        return tensor
    elif isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            return tensor
        # It is tempting to just do tensor.requires_grad = True here, but that will lead to trouble
        return tensor.detach().requires_grad_(True)
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
