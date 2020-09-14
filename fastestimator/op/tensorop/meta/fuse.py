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
import torch

from fastestimator.network import BaseNetwork
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)
Model = TypeVar('Model', tf.keras.Model, torch.nn.Module)


@traceable()
class Fuse(TensorOp):
    """Run a sequence of TensorOps one or more times as a single Op.

    Args:
        ops: A sequence of TensorOps to run. They must all share the same mode. It also doesn't support scheduled ops at
            the moment, though the subnet itself may be scheduled.
        repeat: How many times to repeat the sequence of `tensor_ops`.

    Raises:
        ValueError: If `repeat` or `ops` are invalid.
    """
    def __init__(self, ops: Union[TensorOp, List[TensorOp]], repeat: int = 1) -> None:
        if repeat < 1:
            raise ValueError(f"Fuse requires repeat to be >= 1, but got {repeat}")
        self.repeat = repeat
        ops = to_list(ops)
        if len(ops) < 1:
            raise ValueError("Fuse requires at least one op")
        inputs = []
        outputs = []
        mode = ops[0].mode
        self.last_retain_idx = None
        self.retain_graph = None
        self.models = set()
        self.loss_keys = set()
        for idx, op in enumerate(ops):
            if op.mode != mode:
                raise ValueError(f"All Fuse ops must share the same mode, but got {mode} and {op.mode}")
            for inp in op.inputs:
                if inp not in inputs and inp not in outputs:
                    inputs.append(inp)
            for out in op.outputs:
                if out not in outputs:
                    outputs.append(out)
            if op.fe_retain_graph(True) is not None:  # Set all of the internal ops to retain since might loop them
                self.retain_graph, self.last_retain_idx = (True, idx)
            self.models |= op.get_fe_models()
            self.loss_keys |= op.get_fe_loss_keys()
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.ops = ops

    def build(self, framework: str) -> None:
        for op in self.ops:
            op.build(framework)

    def get_fe_models(self) -> Set[Model]:
        return self.models

    def get_fe_loss_keys(self) -> Set[str]:
        return self.loss_keys

    def fe_retain_graph(self, retain: Optional[bool] = None) -> Optional[bool]:
        if retain is not None:
            self.retain_graph = retain
        return self.retain_graph

    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        for i in range(self.repeat - 1):
            # Perform n-1 rounds with all ops having retain_graph == True
            BaseNetwork._forward_batch(data, state, self.ops)
        # Adjust the final op's retain_graph if necessary
        if self.last_retain_idx is not None:
            self.ops[self.last_retain_idx].fe_retain_graph(self.retain_graph)
        # Final round of ops
        BaseNetwork._forward_batch(data, state, self.ops)
        # Set the last op retain_graph back to True for later rounds
        if self.last_retain_idx is not None:
            self.ops[self.last_retain_idx].fe_retain_graph(True)
        return [data[key] for key in self.outputs]
