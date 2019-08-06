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
from fastestimator.pipeline.filter import TensorFilter
from itertools import chain


class TensorOp:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        return data


class NumpyOp:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data):
        return data


def flatten_operation(ops):
    if not isinstance(ops, list):
        ops = [ops]
    for idx, op in enumerate(ops):
        if not isinstance(op, list):
            ops[idx] = [op]
    ops = list(chain.from_iterable(ops))
    return ops


def get_op_from_mode(ops, current_mode):
    selected_ops = []
    for op in ops:
        op_mode = op.mode
        if not isinstance(op_mode, list):
            op_mode = [op_mode]
        if None in op_mode or current_mode in op_mode:
            selected_ops.append(op)
    return selected_ops


def verify_ops(ops, class_name):
    inheritage = {"RecordWriter": NumpyOp,
                  "Pipeline": (TensorOp, TensorFilter),
                  "Network": TensorOp}
    inheritage_class = inheritage[class_name]
    assert ops[0].inputs, "must provide inputs for the operation '{}' in '{}'".format(type(ops[0]).__name__, class_name)
    assert ops[-1].outputs, "must provide outputs for the operation '{}' in '{}'".format(type(ops[-1]).__name__, class_name)
    inputs = ops[0].inputs
    for idx, op in enumerate(ops):
        assert isinstance(op, inheritage_class), "operation '{}' in class '{}' doesn't have correct inheritage".format(type(op).__name__, class_name)
        if idx + 1 < len(ops) and ops[idx+1].inputs:
            new_inputs = ops[idx+1].inputs
            if new_inputs and new_inputs != inputs:
                assert op.outputs, "must provide outputs for the operation '{}' in class '{}', otherwise the result will be lost".format(type(op).__name__, class_name)