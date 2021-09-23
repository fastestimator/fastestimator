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
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.get_gradient import get_gradient
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


@traceable()
class GradientOp(TensorOp):
    """Return the gradients of finals w.r.t. inputs.

    Args:
        finals: The tensor(s) to compute gradients from.
        outputs: The key(s) under which to save the gradients.
        inputs: The tensor(s) to compute gradients with respect to, mutually exclusive with `model`.
        model: The model instance to compute gradients with respect to, mutually exclusive with `inputs`.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 finals: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 inputs: Union[None, str, List[str]] = None,
                 model: Union[None, tf.keras.Model, torch.nn.Module] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        inputs = to_list(inputs)
        finals = to_list(finals)
        outputs = to_list(outputs)
        assert bool(model) != bool(inputs), "Must provide either one of 'inputs' or 'model'"
        if model is None:
            assert len(inputs) == len(finals) == len(outputs), \
                "GradientOp requires the same number of inputs, finals, and outputs"
        else:
            assert isinstance(model, (tf.keras.Model, torch.nn.Module)), "Unrecognized model format"
            assert len(finals) == len(outputs), "GradientOp requires the same number of finals, and outputs"
        inputs.extend(finals)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.model = model
        self.retain_graph = True

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        self.framework = framework

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        results = []
        if self.model is None:
            initials = data[:len(data) // 2]
            finals = data[len(data) // 2:]
            for idx, (initial, final) in enumerate(zip(initials, finals)):
                retain_graph = self.retain_graph or not idx == len(finals) - 1
                results.append(get_gradient(final, initial, tape=state['tape'], retain_graph=retain_graph))
        else:
            finals = data
            if self.framework == "tf":
                trainable_params = self.model.trainable_variables
                for idx, final in enumerate(finals):
                    gradient = get_gradient(final, trainable_params, tape=state['tape'])
                    results.append(gradient)
            elif self.framework == "torch":
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                for idx, final in enumerate(finals):
                    # get_gradinet
                    retain_graph = self.retain_graph or not idx == len(finals) - 1
                    gradient = get_gradient(final, trainable_params, retain_graph=retain_graph)
                    results.append(gradient)
            else:
                raise ValueError(f"Unrecognized framework {self.framework}")

        return results
