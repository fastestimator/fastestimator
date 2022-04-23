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
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data.dataloader import default_collate

from fastestimator.backend._to_tensor import to_tensor
from fastestimator.op.op import Op, get_inputs_by_op, write_outputs_by_op
from fastestimator.util.data import FilteredData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import pad_batch

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
                state: Dict[str, Any]) -> Union[None, FilteredData, np.ndarray, List[np.ndarray]]:
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

    def forward_batch(self,
                      data: Union[np.ndarray, List[np.ndarray]],
                      state: Dict[str, Any]) -> Union[None, FilteredData, np.ndarray, List[np.ndarray]]:
        """A method which will be invoked in order to transform a batch of data.

        This method will be invoked on batches of data during network postprocessing. It should expect to receive
        batched data and should itself return batched data.

        Args:
            data: The arrays from the data dictionary corresponding to whatever keys this Op declares as its `inputs`.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after applying whatever transform this Op is responsible for. It will be written into the data
            dictionary based on whatever keys this Op declares as its `outputs`.
        """
        if isinstance(data, list):
            data = [elem for elem in map(list, zip(*data))]
        else:
            data = [elem for elem in data]
        results = [self.forward(elem, state) for elem in data]
        if self.out_list:
            results = [np.array(col) for col in [[row[i] for row in results] for i in range(len(results[0]))]]
        else:
            results = np.array(results)
        return results


@traceable()
class Batch(NumpyOp):
    """Convert data instances into a batch of data.

    Only one instance of a Batch Op can be present for a given epoch/mode/ds_id combination. Any Ops after this one will
    operate on batches of data rather than individual instances (using their batch_forward methods).

    Args:
        batch_size: The batch size to use. If set, this will override any value specified by the Pipeline, allowing
            control of the batch size on a per-mode and per-ds_id level. Note that this value will be ignored when using
            a BatchDataset (or any dataset which decides on its own batch configuration).
        drop_last: Whether to drop the last batch if the last batch is incomplete. Note that setting this to True when
            using a BatchDataset (or any dataset which decides on its own batch configuration) won't do anything.
        pad_value: The padding value if batch padding is needed. None indicates that no padding is needed. Mutually
            exclusive with `collate_fn`.
        collate_fn: A function to merge a list of data elements into a batch of data. Mutually exclusive with
            `pad_value`.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 batch_size: Optional[int] = None,
                 drop_last: bool = False,
                 pad_value: Optional[Union[int, float]] = None,
                 collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(mode=mode, ds_id=ds_id)
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise ValueError(f"batch_size must be an integer, but got {type(batch_size)}")
            if batch_size < 0:
                raise ValueError("batch_size must be non-negative")
        self.batch_size = batch_size
        self.drop_last = drop_last
        if pad_value is not None and collate_fn is not None:
            raise ValueError("Provide either a pad_value or collate_fn, but not both")
        self._pad_value = pad_value
        self.collate_fn = collate_fn
        if pad_value is not None:
            self.collate_fn = self._pad_batch_collate
        if self.collate_fn is None:
            # Note that this might get ignored in favor of default_convert inside the FEDataLoader if it looks like the
            # user really doesn't want stuff to be batched.
            self.collate_fn = default_collate

    def _pad_batch_collate(self, batch: List[MutableMapping[str, Any]]) -> Dict[str, Any]:
        """A collate function which pads a batch of data.

        Args:
            batch: The data to be batched and collated.

        Returns:
            A padded and collated batch of data.
        """
        pad_batch(batch, self._pad_value)
        return default_collate(batch)


@traceable()
class Delete(NumpyOp):
    """Delete key(s) and their associated values from the data dictionary.

    The system has special logic to detect instances of this Op and delete its `inputs` from the data dictionary.

    Args:
        keys: Existing key(s) to be deleted from the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
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

    def forward_batch(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> None:
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

    def forward_batch(self, data: Union[Tensor, List[Tensor]], state: Dict[str,
                                                                           Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.forward(data, state)


@traceable()
class RemoveIf(NumpyOp):
    """An Operator which will remove a datapoint from the pipeline if the given criterion is satisfied.

    Args:
        fn: A function taking any desired `inputs` and returning a boolean. If the return value is true, the current
            datapoint (or batch if using a batch dataset) will be removed and another will take its place.
        replacement: Whether to replace the filtered element with another (thus maintaining the number of steps in an
            epoch but potentially increasing data repetition) or else shortening the epoch by the number of filtered
            data points (fewer steps per epoch than expected, but no extra data repetition). Either way, the number of
            data points within an individual batch will remain the same. Even if `replacement` is true, data will not be
            repeated until all of the given epoch's data has been traversed (except for at most 1 batch of data which
            might not appear until after the re-shuffle has occurred).
        inputs: Key(s) from which to retrieve data from the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 fn: Callable[..., bool],
                 replacement: bool = True,
                 inputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, mode=mode, ds_id=ds_id)
        self.filter_fn = fn
        self.in_list = True
        self.replacement = replacement

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> Optional[FilteredData]:
        if self.filter_fn(*data):
            return FilteredData(replacement=self.replacement)
        return None

    def forward_batch(self, data: Union[Tensor, List[Tensor]], state: Dict[str,
                                                                           Any]) -> Optional[FilteredData]:
        return self.forward(data, state)


def forward_numpyop(ops: List[NumpyOp],
                    data: MutableMapping[str, Any],
                    state: Dict[str, Any],
                    batched: Optional[str] = None) -> Optional[FilteredData]:
    """Call the forward function for list of NumpyOps, and modify the data dictionary in place.

    Args:
        ops: A list of NumpyOps to execute.
        data: The data dictionary.
        state: Information about the current execution context, ex. {"mode": "train"}. Must contain at least the mode.
        batched: Whether the `data` is batched or not. If it is batched, provide the string ('tf', 'torch', or 'np')
            indicating which type of tensors the batch contains.
    """
    if not ops:
        # Shortcut to prevent wasting time in to_tensor calls if there aren't any ops
        return None
    if batched:
        # Cast data to Numpy before performing batch forward
        for key, val in data.items():
            data[key] = to_tensor(val, target_type='np')
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
        if isinstance(op_data, FilteredData):
            return op_data
        if isinstance(op, Delete):
            for key in op.inputs:
                del data[key]
        if op.outputs:
            write_outputs_by_op(op, data, op_data)
    if batched:
        # Cast data back to original tensor type after performing batch forward
        for key, val in data.items():
            data[key] = to_tensor(val, target_type=batched, shared_memory=True)
    return None
