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
import pdb
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.util.util import to_list

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class Op:
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        if isinstance(inputs, Iterable) and not isinstance(inputs, str):
            self.inputs = list(inputs)
        else:
            self.inputs = inputs
        if isinstance(outputs, Iterable) and not isinstance(outputs, str):
            self.outputs = list(outputs)
        else:
            self.outputs = outputs
        if isinstance(mode, Iterable) and not isinstance(mode, str):
            self.mode = set(mode)
        else:
            self.mode = mode


class TensorOp(Op):
    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        return data


class NumpyOp(Op):
    def forward(self, data: Union[np.ndarray, List[np.ndarray], str, List[str]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return data


OpType = TypeVar('OpType', Op, NumpyOp, TensorOp)


def get_ops_by_mode(ops: Iterable[OpType], mode: str) -> List[OpType]:
    selected_ops = []
    for op in ops:
        op_mode = op.mode
        if not isinstance(op_mode, list):
            op_mode = [op_mode]
        if None in op_mode or mode in op_mode:
            selected_ops.append(op)
    return selected_ops


def get_inputs_by_op(op: Op, store: Mapping[str, Any], default: Optional[Any] = None) -> Any:
    data = default
    if op.inputs:
        if isinstance(op.inputs, Callable):
            data = op.inputs()
        else:
            data = get_inputs_by_key(store, op.inputs)
    return data


def get_inputs_by_key(store: Mapping[str, Any], inputs_key: str) -> Any:
    if isinstance(inputs_key, list):
        data = [store[key] for key in inputs_key]
    else:
        data = store[inputs_key]
    return data


def write_outputs_by_key(store: Mapping[str, Any], output: Any, outputs_key: str):
    if isinstance(outputs_key, str):
        store[outputs_key] = output
    else:
        for key, data in zip(outputs_key, output):
            store[key] = data
