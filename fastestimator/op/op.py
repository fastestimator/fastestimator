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
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch

from fastestimator.schedule import Scheduler
from fastestimator.util.util import correct_modes, to_list, to_set

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


class Op:
    inputs: List[Union[str, Callable]]
    outputs: List[str]
    mode: Set[str]
    in_list: bool  # Whether inputs should be presented as a list or an individual value
    out_list: bool  # Whether outputs will be returned as a list or an individual value

    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None):
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.mode = correct_modes(to_set(mode))
        self.in_list = not isinstance(inputs, (str, Callable))
        self.out_list = not isinstance(outputs, str)


class TensorOp(Op):
    def forward(self, data: Union[Tensor, List[Tensor]], state: Dict[str, Any]) -> Union[Tensor, List[Tensor]]:
        return data


class NumpyOp(Op):
    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return data


OpType = TypeVar('OpType', bound=Op)


def get_current_ops(ops: Iterable[Union[OpType, Scheduler[OpType]]], mode: str, epoch: int = 0) -> List[OpType]:
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
        data = [store[key] if not isinstance(key, Callable) else key() for key in op.inputs]
        if not op.in_list:
            data = data[0]
    return data


def write_outputs_by_op(op: Op, store: MutableMapping[str, Any], outputs: Any):
    if not op.out_list:
        outputs = [outputs]
    for key, data in zip(op.outputs, outputs):
        store[key] = data
