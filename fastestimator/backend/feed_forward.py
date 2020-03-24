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

from fastestimator.backend.to_tensor import to_tensor

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def feed_forward(model: Union[tf.keras.Model, torch.nn.Module], x: Tensor, training: bool = True) -> Tensor:
    if isinstance(model, tf.keras.Model):
        if not isinstance(x, tf.Tensor):
            x = to_tensor(x, "tf")
        x = model(x, training=training)
    elif isinstance(model, torch.nn.Module):
        model.train(mode=training)
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x, "torch")
        x = model(x)
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
    return x
