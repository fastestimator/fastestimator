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
from typing import Any, Dict, Iterable, List, TypeVar, Union

import tensorflow as tf
import torch
from tensorflow.python.framework import ops as tfops

from fastestimator.backend.update_model import update_model
from fastestimator.op.op import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the loss

    Args:
        model: model instance compiled by fe.build
        loss_name: the name of loss
        mode: The mode or modes for which to execute
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 loss_name: str,
                 mode: Union[None, str, Iterable[str]] = "train"):
        super().__init__(inputs=loss_name, outputs=None, mode=mode)
        self.model = model
        self.retain_graph = False
        if not hasattr(self.model, "loss_name"):
            self.model.loss_name = {loss_name}
        else:
            self.model.loss_name.add(loss_name)

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]):
        if state["warmup"]:
            if isinstance(self.model, tf.keras.Model):
                with tfops.init_scope():
                    _ = self.model.current_optimizer.iterations
                    self.model.current_optimizer._create_hypers()
                    self.model.current_optimizer._create_slots(self.model.trainable_variables)
        else:
            update_model(self.model, data, tape=state['tape'], retain_graph=self.retain_graph)
