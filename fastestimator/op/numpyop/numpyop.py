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
from typing import Any, Dict, Iterable, List, MutableMapping, Union

import numpy as np

from fastestimator.op.op import Op, get_inputs_by_op, write_outputs_by_op


class NumpyOp(Op):
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return data


class Delete(NumpyOp):
    """Delete the key, value pairs in data dict.

        Args:
            keys: Existing key(s) to be deleted in data dict.
    """
    def __init__(self, keys: Union[str, List[str]], mode: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=keys, mode=mode)

    def forward(self, data: Union[np.ndarray, List[np.ndarray]], state: Dict[str, Any]) -> None:
        pass


def forward_numpyop(ops: List[NumpyOp], data: MutableMapping[str, Any], mode: str):
    """call the forward for list of numpy Ops, modify the data in place

    Args:
        ops: list of NumpyOps
        data: data dictionary
        mode: the current execution mode
    """
    op_data = None
    for op in ops:
        op_data = get_inputs_by_op(op, data, op_data)
        op_data = op.forward(op_data, {"mode": mode})
        if isinstance(op, Delete):
            for key in op.inputs:
                del data[key]
        if op.outputs:
            write_outputs_by_op(op, data, op_data)
