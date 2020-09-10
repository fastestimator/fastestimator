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
from typing import Any, Dict, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list


@traceable()
class Fuse(NumpyOp):
    """Run a sequence of TensorOps one or more times as a single Op.

    Args:
        ops: A sequence of TensorOps to run. They must all share the same mode. It also doesn't support scheduled ops at
            the moment, though the subnet itself may be scheduled.
        repeat: How many times to repeat the sequence of `tensor_ops`.

    Raises:
        ValueError: If `repeat` or `ops` are invalid.
    """
    def __init__(self, ops: Union[NumpyOp, List[NumpyOp]], repeat: int = 1) -> None:
        if repeat < 1:
            raise ValueError(f"Subnet requires repeat to be >= 1, but got {repeat}")
        self.repeat = repeat
        ops = to_list(ops)
        if len(ops) < 1:
            raise ValueError("Subnet requires at least one op")
        inputs = []
        outputs = []
        mode = ops[0].mode
        for op in ops:
            if op.mode != mode:
                raise ValueError(f"All subnet ops must share the same mode, but got {mode} and {op.mode}")
            for inp in op.inputs:
                if inp not in inputs and inp not in outputs:
                    inputs.append(inp)
            for out in op.outputs:
                if out not in outputs:
                    outputs.append(out)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.ops = ops

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        for i in range(self.repeat):
            forward_numpyop(self.ops, data, state["mode"])
        return [data[key] for key in self.outputs]
