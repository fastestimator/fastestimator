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

from fastestimator.backend.reduce_loss import reduce_loss


def update_model(model: Union[tf.keras.Model, torch.nn.Module],
                 loss: Union[tf.Tensor, torch.Tensor],
                 tape: Optional[tf.GradientTape] = None):
    loss = reduce_loss(loss)
    if isinstance(model, tf.keras.Model):
        with tape.stop_recording():
            gradients = tape.gradient(loss, model.trainable_variables)
            model.current_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    elif isinstance(model, torch.nn.Module):
        loss.backward(retain_graph=True)
        model.current_optimizer.step()
        model.current_optimizer.zero_grad()
    else:
        raise ValueError("Unrecognized model instance {}".format(type(model)))
