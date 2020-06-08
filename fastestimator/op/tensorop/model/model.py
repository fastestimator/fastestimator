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

from fastestimator.backend.feed_forward import feed_forward
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class ModelOp(TensorOp):
    """This class performs forward passes of a neural network over batch data to generate predictions.

    Args:
        model: A model compiled by fe.build.
        inputs: String key of input training data.
        outputs: String key under which to store predictions.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        trainable: Indicates whether the model should have its weights tracked for update.
    """
    def __init__(self,
                 model: Union[tf.keras.Model, torch.nn.Module],
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 trainable: bool = True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        assert hasattr(model, "fe_compiled"), "must use fe.build to compile the model before use"
        self.model = model
        self.trainable = trainable

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        training = state['mode'] == "train" and self.trainable
        data = feed_forward(self.model, data, training=training)
        return data
