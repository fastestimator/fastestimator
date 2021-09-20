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
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.op import Op, get_inputs_by_op, write_outputs_by_op
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor, np.ndarray)


@traceable()
class NumpyOp(Op):
    """An Operator class which takes and returns numpy data.

    These Operators are used in fe.Pipeline to perform data pre-processing / augmentation. They may also be used in
    fe.Network to perform postprocessing on data.

    Args:
        inputs: Key(s) from which to retrieve data from the data dictionary.
        outputs: Key(s) under which to write the outputs of this Op back to the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        # in_place_edits tracks whether the .forward() method of this op will perform in-place edits of numpy arrays.
        # This is inferred automatically by the system and is used for memory management optimization. If you are
        # developing a NumpyOp which does in-place edits, the best practice is to set this to True in your init method.
        self.in_place_edits = False

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

    def forward_batch(self, data: Union[Tensor, List[Tensor]], state: Dict[str,
                                                                           Any]) -> Union[np.ndarray, List[np.ndarray]]:
        """A method which will be invoked in order to transform a batch of data.

        This method will be invoked on batches of data during network postprocessing. Note that the inputs may be numpy
        arrays or TF/Torch tensors. Outputs are expected to be Numpy arrays, though this is not enforced. Developers
        should probably not need to override this implementation unless they are building an op specifically intended
        for postprocessing.

        Args:
            data: The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data
            dictionary based on whatever keys this Op declares as its `outputs`.
        """
        if isinstance(data, List):
            data = [to_number(elem) for elem in data]
            batch_size = data[0].shape[0]
            data = [[elem[i] for elem in data] for i in range(batch_size)]
        else:
            data = to_number(data)
            data = [data[i] for i in range(data.shape[0])]
        results = [self.forward(elem, state) for elem in data]
        if self.out_list:
            results = [np.array(col) for col in [[row[i] for row in results] for i in range(len(results[0]))]]
        else:
            results = np.array(results)
        return results


@traceable()
class Delete(NumpyOp):
    """Delete key(s) and their associated values from the data dictionary.

    The system has special logic to detect instances of this Op and delete its `inputs` from the data dictionary.

    Args:
        keys: Existing key(s) to be deleted from the data dictionary.
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 keys: Union[str, List[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=keys, mode=mode, ds_id=ds_id)

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
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 fn: Callable,
                 inputs: Union[None, str, Iterable[str]] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.fn = fn
        self.in_list = True

    def set_rua_level(self, magnitude_coef: float) -> None:
        """A method which will be invoked by the RUA Op to adjust the augmentation intensity.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.fn(*data)


def forward_numpyop(ops: List[NumpyOp], data: MutableMapping[str, Any], state: Dict[str, Any],
                    batched: bool = False) -> None:
    """Call the forward function for list of NumpyOps, and modify the data dictionary in place.

    Args:
        ops: A list of NumpyOps to execute.
        data: The data dictionary.
        state: Information about the current execution context, ex. {"mode": "train"}. Must contain at least the mode.
        batched: Whether the `data` is batched or not.
    """
    for op in ops:
        op_data = get_inputs_by_op(op, data, copy_on_write=op.in_place_edits)
        try:
            op_data = op.forward_batch(op_data, state) if batched else op.forward(op_data, state)
        except ValueError as err:
            if err.args[0] == 'assignment destination is read-only':
                # If the numpy error text changes we'll need to make adjustments in the future
                op.in_place_edits = True
                op_data = get_inputs_by_op(op, data, copy_on_write=op.in_place_edits)
                op_data = op.forward_batch(op_data, state) if batched else op.forward(op_data, state)
            else:
                raise err
        if isinstance(op, Delete):
            for key in op.inputs:
                del data[key]
        if op.outputs:
            write_outputs_by_op(op, data, op_data)
