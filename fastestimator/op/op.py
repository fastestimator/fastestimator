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


class TensorOp:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data, state):
        return data


class NumpyOp:
    def __init__(self, inputs=None, outputs=None, mode=None):
        self.inputs = inputs
        self.outputs = outputs
        self.mode = mode

    def forward(self, data, state):
        return data


def get_op_from_mode(ops, current_mode):
    selected_ops = []
    for op in ops:
        assert hasattr(op, "mode"), "Operation: {} has no mode attribute".format(op)
        op_mode = op.mode
        if not isinstance(op_mode, list):
            op_mode = [op_mode]
        if None in op_mode or current_mode in op_mode:
            selected_ops.append(op)
    return selected_ops


def get_inputs_by_op(op, store, default=None):
    data = default
    if op.inputs:
        if hasattr(op.inputs, "__call__"):
            data = op.inputs()
        else:
            data = get_inputs_by_key(store, op.inputs)
    return data


def get_inputs_by_key(store, inputs_key):
    if isinstance(inputs_key, list):
        data = [store[key] for key in inputs_key]
    elif isinstance(inputs_key, tuple):
        data = tuple([store[key] for key in inputs_key])
    else:
        data = store[inputs_key]
    return data


def write_outputs_by_key(store, output, outputs_key):
    if isinstance(outputs_key, str):
        store[outputs_key] = output
    else:
        for key, data in zip(outputs_key, output):
            store[key] = data
    return store
