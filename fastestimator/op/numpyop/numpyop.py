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
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Union

import numpy as np

from fastestimator.op.op import Op, get_inputs_by_op, write_outputs_by_op
from fastestimator.util.traceability_util import traceable


@traceable()
class NumpyOp(Op):
    """An Operator class which takes and returns numpy data.

    These Operators are used in fe.Pipeline to perform data pre-processing / augmentation.
    """
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        """A method which will be invoked in order to transform data.

        This method will be invoked on individual elements of data before any batching / axis expansion is performed.

        Args:
            data: The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data
            dictionary based on whatever keys this Op declares as its `outputs`.
        """
        return data


@traceable()
class Delete(NumpyOp):
    """Delete key(s) and their associated values from the data dictionary.

    The system has special logic to detect instances of this Op and delete its `inputs` from the data dictionary.

    Args:
        keys: Existing key(s) to be deleted from the data dictionary.
    """
    def __init__(self, keys: Union[str, List[str]], mode: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=keys, mode=mode)

    def forward(self, data: Union[np.ndarray, List[np.ndarray]], state: Dict[str, Any]) -> None:
        pass


@traceable()
class LambdaOp(NumpyOp):
    """An Operator that performs any specified function as forward function.

    Args:
        fn: The function to be executed.
        inputs: Key(s) from which to retrieve data from the data dictionary.
        outputs: Key(s) under which to write the outputs of this Op back to the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 fn: Callable,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.fn = fn
        self.in_list = True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.fn(*data)


def forward_numpyop(ops: List[NumpyOp], data: MutableMapping[str, Any], mode: str) -> None:
    """Call the forward function for list of NumpyOps, and modify the data dictionary in place.

    Args:
        ops: A list of NumpyOps to execute.
        data: The data dictionary.
        mode: The current execution mode ("train", "eval", "test", or "infer").
    """
    for op in ops:
        op_data = get_inputs_by_op(op, data)
        op_data = op.forward(op_data, {"mode": mode})
        if isinstance(op, Delete):
            for key in op.inputs:
                del data[key]
        if op.outputs:
            write_outputs_by_op(op, data, op_data)
