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
import tensorflow as tf

from fastestimator.network.model import ModelOp
from fastestimator.util.op import get_op_from_mode, verify_ops
from fastestimator.util.schedule import Scheduler
from fastestimator.util.util import NonContext


class Network:
    def __init__(self, ops):
        if not isinstance(ops, list):
            ops = [ops]
        self.ops = ops
        self.model_schedule = {}
        self.op_schedule = {}
        self.current_epoch_ops = {}
        self.current_epoch_model = {}

    def _prepare(self, mode_list):
        for mode in mode_list:
            signature_epoch, mode_ops = self._get_signature_epoch(mode)
            epoch_ops_map = {}
            epoch_model_map = {}
            for epoch in signature_epoch:
                epoch_ops = []
                epoch_model = []
                #generate ops for specific mode and epoch
                for op in mode_ops:
                    if isinstance(op, Scheduler):
                        scheduled_op = op.get_current_value(epoch)
                        if scheduled_op:
                            epoch_ops.append(scheduled_op)
                    else:
                        epoch_ops.append(op)
                #check the ops
                verify_ops(epoch_ops, "Network")
                #create model list
                for op in epoch_ops:
                    if isinstance(op, ModelOp) and op.model not in epoch_model:
                        epoch_model.append(op.model)
                assert epoch_model, "Network has no model for epoch {}".format(epoch)
                epoch_ops_map[epoch] = epoch_ops
                epoch_model_map[epoch] = epoch_model
            self.op_schedule[mode] = Scheduler(epoch_dict=epoch_ops_map)
            self.model_schedule[mode] = Scheduler(epoch_dict=epoch_model_map)

    def _get_signature_epoch(self, mode):
        signature_epoch = [0]
        mode_ops = get_op_from_mode(self.ops, mode)
        for op in mode_ops:
            if isinstance(op, Scheduler):
                signature_epoch.extend(op.keys)
        return list(set(signature_epoch)), mode_ops

    def load_epoch(self, epoch, mode):
        self.current_epoch_ops[mode] = self.op_schedule[mode].get_current_value(epoch)
        self.current_epoch_model[mode] = self.model_schedule[mode].get_current_value(epoch)

    def run_step(self, batch, state, warm_up=False):
        losses = ()
        mode = state["mode"]
        ops = self.current_epoch_ops[mode]
        model_list = self.current_epoch_model[mode]
        num_model = len(model_list)
        # use gradient tape for train, otherwise use a dummy tape(to save computation)
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward(batch, state, ops)
            for idx in range(num_model):
                losses += self._loss(model_list[idx], batch, state),
        # update model only for train mode
        if mode == "train":
            for idx in range(num_model):
                gradients = tape.gradient(losses[idx], model_list[idx].trainable_variables)
                model_list[idx].optimizer.apply_gradients(zip(gradients, model_list[idx].trainable_variables))
        del state['tape']
        del tape
        return losses

    def _loss(self, model, batch, state):
        op = model.loss
        data = None
        if op.inputs:
            if hasattr(op.inputs, "__call__"):
                data = op.inputs()
            else:
                data = self._get_inputs_from_key(batch, op.inputs)
        data = op.forward(data, state)
        if op.outputs:
            self._write_outputs_to_key(data, batch, op.outputs)
        return data

    def _forward(self, batch, state, ops):
        for op in ops:
            if op.inputs:
                if hasattr(op.inputs, "__call__"):
                    data = op.inputs()
                else:
                    data = self._get_inputs_from_key(batch, op.inputs)
            data = op.forward(data, state)
            if op.outputs:
                self._write_outputs_to_key(data, batch, op.outputs)

    def _get_inputs_from_key(self, batch, inputs_key):
        if isinstance(inputs_key, list):
            data = [batch[key] for key in inputs_key]
        elif isinstance(inputs_key, tuple):
            data = tuple([batch[key] for key in inputs_key])
        else:
            data = batch[inputs_key]
        return data

    def _write_outputs_to_key(self, data, batch, outputs_key):
        if isinstance(outputs_key, str):
            batch[outputs_key] = data
        else:
            for key, value in zip(outputs_key, data):
                batch[key] = value
