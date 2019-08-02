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
from fastestimator.util.op import flatten_operation, get_op_from_mode, verify_ops
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp
from fastestimator.util.util import NonContext
import tensorflow as tf


class Network:
    def __init__(self, ops=None):
        self.ops = ops
        self.model_list = []
        self.mode_ops = {}
        if self.ops:
            self._verify_inputs()

    def _verify_inputs(self):
        self.ops = flatten_operation(self.ops)
        for op in self.ops:
            if isinstance(op, ModelOp) and op.model not in self.model_list:
                self.model_list.append(op.model)
        self.num_model = len(self.model_list)
        assert self.num_model > 0, "Network contains no model"

    def _check_ops(self, mode):
        self.mode_ops[mode] = get_op_from_mode(self.ops, mode)
        verify_ops(self.mode_ops[mode], "Network")

    def run_step(self, batch, mode, epoch):
        losses = ()
        # use gradient tape for train, otherwise use a dummy tape(to save computation)
        with tf.GradientTape(persistent=True) if mode=="train" else NonContext() as tape:
            prediction = self._forward(batch, mode, epoch)
            for idx in range(self.num_model):
                losses += self.model_list[idx].loss.calculate_loss(batch, prediction),
        #update model only for train mode
        if mode == "train":
            for idx in range(self.num_model):
                gradients = tape.gradient(losses[idx], self.model_list[idx].trainable_variables)
                self.model_list[idx].optimizer.apply_gradients(zip(gradients,self.model_list[idx].trainable_variables))
        del tape
        return prediction, losses
    
    def _forward(self, batch, mode, epoch):
        prediction = {}
        for op in self.mode_ops[mode]:
            if op.inputs:
                if hasattr(op.inputs, "__call__"):
                    data = op.inputs()
                else:
                    data = self._get_inputs_from_key(batch, prediction, op.inputs)
            if isinstance(op, ModelOp):
                data = op.forward(data, mode)
            else:
                data = op.forward(data)
            if op.outputs:
                prediction = self._write_outputs_to_key(data, prediction, op.outputs)
        return prediction

    def _get_inputs_from_key(self, batch, prediction, inputs_key):
        if isinstance(inputs_key, list):
            data = [self._select_data_from_key(batch, prediction, key) for key in inputs_key]
        elif isinstance(inputs_key, tuple):
            data = tuple([self._select_data_from_key(batch, prediction, key) for key in inputs_key])
        else:
            data = self._select_data_from_key(batch, prediction, inputs_key)
        return data
    
    def _select_data_from_key(self, batch, prediction, key):
        if key in prediction:
            data = prediction[key]
        else:
            data = batch[key]
        return data

    def _write_outputs_to_key(self, data, prediction, outputs_key):
        if isinstance(outputs_key, str):
            prediction[outputs_key] = data
        else:
            for key, value in zip(outputs_key, data):
                prediction[key] = value
        return prediction