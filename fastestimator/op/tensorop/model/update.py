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
from fastestimator.op.tensorop.tensorop import TensorOp

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class UpdateOp(TensorOp):
    """This class performs updates to a model's weights based on the loss.

    Args:
        model: Model instance compiled by fe.build.
        loss_name: The name of loss.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 loss_name: str,
                 mode: Union[None, str, Iterable[str]] = "train"):
        super().__init__(inputs=loss_name, outputs=None, mode=mode)
        self.model = model
        self.retain_graph = False
        self.weight_decay = isinstance(self.model, tf.keras.Model) and self.model.losses
        if not hasattr(self.model, "loss_name"):
            self.model.loss_name = {loss_name}
        else:
            self.model.loss_name.add(loss_name)

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]):
        if not state["warmup"]:
            if self.weight_decay:
                data = data + tf.reduce_sum(self.model.losses)
            update_model(self.model, data, tape=state['tape'], retain_graph=self.retain_graph)
