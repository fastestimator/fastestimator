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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, TypeVar, Union, Set, MutableMapping

import numpy as np
import tensorflow as tf
import torch

from fastestimator.schedule import Scheduler
from fastestimator.util.util import to_list, to_set

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class Op:
    inputs: List[str]
    outputs: List[str]
    mode: Set[str]

    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = to_set(mode)


class TensorOp(Op):
    def forward(self, data: List[Tensor], state: Dict[str, Any]) -> List[Tensor]:
        return data


class NumpyOp(Op):
    def forward(self, data: Union[List[np.ndarray], List[str]], state: Dict[str, Any]) -> List[np.ndarray]:
        return data


OpType = TypeVar('OpType', Op, NumpyOp, TensorOp)


def get_current_ops(ops: Iterable[OpType], mode: str, epoch: int = 0) -> List[OpType]:
    selected_ops = []
    for op in ops:
        if isinstance(op, Scheduler):
            op = op.get_current_value(epoch)
        if op and (not op.mode or mode in op.mode):
            selected_ops.append(op)
    return selected_ops


def get_inputs_by_op(op: Op, store: Mapping[str, Any], default: Optional[Any] = None) -> Any:
    data = default
    if op.inputs:
        if isinstance(op.inputs, Callable):
            data = op.inputs()
        else:
            data = get_inputs_by_keys(store, op.inputs)
    return data


def get_inputs_by_keys(store: Mapping[str, Any], input_keys: List[str]) -> Any:
    return [store[key] for key in input_keys]


def write_outputs_by_keys(store: MutableMapping[str, Any], outputs: List[Any], output_keys: List[str]):
    for key, data in zip(output_keys, outputs):
        store[key] = data
