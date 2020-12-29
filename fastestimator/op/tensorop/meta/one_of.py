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
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch

from fastestimator.backend.cast import cast
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class OneOf(TensorOp):
    """Perform one of several possible TensorOps.

    Args:
        *tensor_ops: A list of ops to choose between with uniform probability.
    """
    def __init__(self, *tensor_ops: TensorOp) -> None:
        inputs = tensor_ops[0].inputs
        outputs = tensor_ops[0].outputs
        mode = tensor_ops[0].mode
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list = tensor_ops[0].in_list
        self.out_list = tensor_ops[0].out_list
        for op in tensor_ops[1:]:
            assert inputs == op.inputs, "All ops within a OneOf must share the same inputs"
            assert self.in_list == op.in_list, "All ops within OneOf must share the same input configuration"
            assert outputs == op.outputs, "All ops within a OneOf must share the same outputs"
            assert self.out_list == op.out_list, "All ops within OneOf must share the same output configuration"
            assert mode == op.mode, "All ops within a OneOf must share the same mode"
        self.ops = tensor_ops
        self.prob_fn = None
        self.invoke_fn = None

    def build(self, framework: str, device: Optional[torch.device] = None) -> None:
        for op in self.ops:
            op.build(framework, device)
        if framework == 'tf':
            self.prob_fn = tfp.distributions.Uniform(low=0, high=len(self.ops))
            self.invoke_fn = lambda idx, data, state: tf.switch_case(idx, [lambda: op.forward(data, state) for op in
                                                                           self.ops])
        elif framework == 'torch':
            self.prob_fn = torch.distributions.uniform.Uniform(low=0, high=len(self.ops))
            self.invoke_fn = lambda idx, data, state: self.ops[idx].forward(data, state)
        else:
            raise ValueError("unrecognized framework: {}".format(framework))

    def get_fe_loss_keys(self) -> Set[str]:
        return set.union(*[op.get_fe_loss_keys() for op in self.ops])

    def get_fe_models(self) -> Set[Model]:
        return set.union(*[op.get_fe_models() for op in self.ops])

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        resp = None
        for op in self.ops:
            resp = resp or op.fe_retain_graph(retain)
        return resp

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'ops': [elem.__getstate__() if hasattr(elem, '__getstate__') else {} for elem in self.ops]}

    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        """Execute a randomly selected op from the list of `numpy_ops`.

        Args:
            data: The information to be passed to one of the wrapped operators.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after application of one of the available numpyOps.
        """
        idx = cast(self.prob_fn.sample(), dtype='int32')
        return self.invoke_fn(idx, data, state)
