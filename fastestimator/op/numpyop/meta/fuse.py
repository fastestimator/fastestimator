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
import inspect
from typing import Any, Dict, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import Batch, Delete, NumpyOp, forward_numpyop
from fastestimator.util.traceability_util import traceable
from fastestimator.util.base_util import to_list


@traceable()
class Fuse(NumpyOp):
    """Run a sequence of NumpyOps as a single Op.

    Args:
        ops: A sequence of NumpyOps to run. They must all share the same mode. It also doesn't support scheduled ops at
            the moment, though the Fuse itself may be scheduled.

    Raises:
        ValueError: If `repeat` or `ops` are invalid.
    """
    def __init__(self, ops: Union[NumpyOp, List[NumpyOp]]) -> None:
        ops = to_list(ops)
        if len(ops) < 1:
            raise ValueError("Fuse requires at least one op")
        inputs = []
        outputs = []
        mode = ops[0].mode
        ds_id = ops[0].ds_id
        for op in ops:
            if isinstance(op, Batch):
                raise ValueError("Cannot nest the Batch op inside of Fuse")
            if op.mode != mode:
                raise ValueError(f"All Fuse ops must share the same mode, but got {mode} and {op.mode}")
            if op.ds_id != ds_id:
                raise ValueError(f"All Fuse ops must share the same ds_id, but got {ds_id} and {op.ds_id}")
            for inp in op.inputs:
                if isinstance(op, Delete) and inp in outputs:
                    outputs.remove(inp)
                elif inp not in inputs and inp not in outputs:
                    inputs.append(inp)
            for out in op.outputs:
                if out not in outputs:
                    outputs.append(out)
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.ops = ops

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'ops': [elem.__getstate__() if hasattr(elem, '__getstate__') else {} for elem in self.ops]}

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).

        Raises:
            AttributeError: If ops don't have a 'set_rua_level' method.
        """
        for op in self.ops:
            if hasattr(op, "set_rua_level") and inspect.ismethod(getattr(op, "set_rua_level")):
                op.set_rua_level(magnitude_coef=magnitude_coef)
            else:
                raise AttributeError(
                    "RUA Augmentations should have a 'set_rua_level' method but it's not present in Op: {}".format(
                        op.__class__.__name__))

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        filtered = forward_numpyop(self.ops, data, state)
        return filtered if filtered else [data[key] for key in self.outputs]

    def forward_batch(self, data: Union[np.ndarray, List[np.ndarray]],
                      state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        filtered = forward_numpyop(self.ops, data, state, batched="np")
        return filtered if filtered else [data[key] for key in self.outputs]
