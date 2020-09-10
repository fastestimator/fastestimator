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
        inputs: The tensor(s) to compute gradients with respect to.
        finals: The tensor(s) to compute gradients from.
        outputs: The key(s) under which to save the gradients.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 finals: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        inputs = to_list(inputs)
        finals = to_list(finals)
        outputs = to_list(outputs)
        assert len(inputs) == len(finals) == len(outputs), \
            "GradientOp requires the same number of inputs, finals, and outputs"
        inputs.extend(finals)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.retain_graph = True

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        initials = data[:len(data) // 2]
        finals = data[len(data) // 2:]
        results = []
        for initial, final in zip(initials, finals):
            results.append(get_gradient(final, initial, tape=state['tape'], retain_graph=self.retain_graph))
        return results
