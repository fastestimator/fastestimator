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
from typing import Optional, Union

import tensorflow as tf
import torch

from fastestimator.backend.get_gradient import get_gradient
from fastestimator.backend.reduce_loss import reduce_loss


def update_model(model: Union[tf.keras.Model, torch.nn.Module],
                 loss: Union[tf.Tensor, torch.Tensor],
                 tape: Optional[tf.GradientTape] = None,
                 retain_graph: bool = True):
    loss = reduce_loss(loss)
    if isinstance(model, tf.keras.Model):
        strategy = tf.distribute.get_strategy()
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            loss = loss / strategy.num_replicas_in_sync
        gradients = get_gradient(loss, model.trainable_variables, tape=tape)
        with tape.stop_recording():
            model.current_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    elif isinstance(model, torch.nn.Module):
        gradients = get_gradient(loss, model.parameters(), retain_graph=retain_graph)
        for gradient, parameter in zip(gradients, model.parameters()):
            parameter.grad = gradient
        model.current_optimizer.step()
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
