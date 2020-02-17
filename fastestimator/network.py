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
from collections import ChainMap, defaultdict

import tensorflow as tf

import fastestimator as fe
from fastestimator.op import get_inputs_by_op, get_op_from_mode, verify_ops, write_outputs_by_key
from fastestimator.op.tensorop import Loss, ModelOp, UpdateOp
from fastestimator.schedule import Scheduler
from fastestimator.util.util import NonContext, flatten_list, to_list, to_set


class Network:
    """A class representing network operations for FastEstimator model training.

    Args:
        ops : Specifies the series of operations for training model
    """
    def __init__(self, ops):

        if not isinstance(ops, list):
            ops = [ops]
        self.ops = ops
        self.model_schedule = {}
        self.op_schedule = {}
        self.model = {}
        self.epoch_losses = []
        self.all_output_keys = set()
        self.stop_training = False
        self.num_devices = 1

    def prepare(self, mode_list):
        """This function constructs the operations necessary for each epoch
        """
        all_output_keys = []
        all_models = []
        for mode in mode_list:
            signature_epoch, mode_ops = self._get_signature_epoch(mode)
            epoch_ops_map = {}
            epoch_model_map = {}
            for epoch in signature_epoch:
                epoch_ops = []
                epoch_model = []
                epoch_model_update = defaultdict(lambda: False)
                # generate ops for specific mode and epoch
                for op in mode_ops:
                    if isinstance(op, Scheduler):
                        scheduled_op = op.get_current_value(epoch)
                        if scheduled_op:
                            epoch_ops.append(scheduled_op)
                    else:
                        epoch_ops.append(op)
                # check the ops
                verify_ops(epoch_ops, "Network")
                # create model list
                for op in epoch_ops:
                    all_output_keys.append(op.outputs)
                    if isinstance(op, ModelOp):
                        if op.model not in epoch_model:
                            epoch_model.append(op.model)
                            epoch_model_update[op.model] = epoch_model_update[op.model]
                        if op.model not in all_models:
                            all_models.append(op.model)
                    if isinstance(op, UpdateOp):
                        epoch_model_update[op.model] = True
                if mode == "train":
                    for model, has_update in epoch_model_update.items():
                        if not has_update:
                            epoch_ops.append(UpdateOp(model=model))
                assert epoch_model, "Network has no model for epoch {}".format(epoch)
                epoch_ops_map[epoch] = epoch_ops
                epoch_model_map[epoch] = epoch_model
            self.op_schedule[mode] = Scheduler(epoch_dict=epoch_ops_map)
            self.model_schedule[mode] = Scheduler(epoch_dict=epoch_model_map)
        self.all_output_keys = set(flatten_list(all_output_keys)) - {None}
        for model in all_models:
            assert model.model_name not in self.model, "duplicated model name: {}".format(model.model_name)
            self.model[model.model_name] = model

    def _get_signature_epoch(self, mode):
        signature_epoch = [0]
        mode_ops = get_op_from_mode(self.ops, mode)
        for op in mode_ops:
            if isinstance(op, Scheduler):
                signature_epoch.extend(op.keys)
        return list(set(signature_epoch)), mode_ops

    def load_epoch(self, epoch, mode):
        """ This function loads stable computational graph for the current epoch.

        Args:
            epoch: Training epoch number
            mode: 'train' or 'eval'

        Returns:
             list of the models, epoch losses
        """
        ops = self.op_schedule[mode].get_current_value(epoch)
        epoch_losses = set()
        for op in ops:
            if isinstance(op, Loss):
                epoch_losses |= to_set(op.outputs)
        self.epoch_losses = to_list(epoch_losses)
        return ops

    def run_step(self, batch, ops, state):
        """Function that calculates the loss and gradients for curent step in training. It also constructs the higher
        level computational graph between the models before the training.

        Args:
            batch : dictionary that contains batch data and predictions from last epoch
            ops : Model operation dictionary that contains 'Inputs','Mode', and 'Outputs'
            state : run time dictionary that contains following keys 'mode' and 'batch size'

        Returns:
            dictionary containing the predictions of current epoch
        """
        prediction = {}
        batch = ChainMap(prediction, batch)
        mode = state["mode"]
        # use gradient tape for train, otherwise use a dummy tape
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward(batch, state, ops)
        del state['tape']
        del tape
        return prediction

    @staticmethod
    def _forward(batch, state, ops):
        data = None
        for op in ops:
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_key(batch, data, op.outputs)


def build(model_def, model_name, optimizer, loss_name, custom_objects=None):
    """build keras model instance in FastEstimator

    Args:
        model_def (function): function definition of tf.keras model or path of model file(h5)
        model_name (str, list, tuple): model name(s)
        optimizer (str, optimizer, list, tuple): optimizer(s)
        loss_name (str, list, tuple): loss name(s)
        custom_objects (dict): dictionary that maps custom

    Returns:
        model: model(s) compiled by FastEstimator
    """
    with fe.distribute_strategy.scope() if fe.distribute_strategy else NonContext():
        if isinstance(model_def, str):
            model = tf.keras.models.load_model(model_def, custom_objects=custom_objects)
        else:
            model = model_def()
        model = to_list(model)
        model_name = to_list(model_name)
        optimizer = to_list(optimizer)
        loss_name = to_list(loss_name)
        assert len(model) == len(model_name) == len(optimizer) == len(loss_name)
        for idx, (m, m_n, o, l_n) in enumerate(zip(model, model_name, optimizer, loss_name)):
            model[idx] = _fe_compile(m, m_n, o, l_n)
    if len(model) == 1:
        model = model[0]
    return model


def _fe_compile(model, model_name, optimizer, loss_name):
    if isinstance(optimizer, str):
        optimizer_fn = {
            'adadelta': tf.optimizers.Adadelta,
            'adagrad': tf.optimizers.Adagrad,
            'adam': tf.optimizers.Adam,
            'adamax': tf.optimizers.Adamax,
            'nadam': tf.optimizers.Nadam,
            'rmsprop': tf.optimizers.RMSprop,
            'sgd': tf.optimizers.SGD
        }
        optimizer = optimizer_fn[optimizer]()
    else:
        assert isinstance(optimizer, tf.optimizers.Optimizer), \
            "must provide provide must provide tf.optimizer.Optimizer instance as optimizer"
    assert isinstance(model_name, str), "model_name must be string"
    assert isinstance(loss_name, str), "loss_name must be string"
    model.model_name = model_name
    model.optimizer = optimizer
    model.loss_name = loss_name
    model.fe_compiled = True
    return model
