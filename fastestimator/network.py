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
from collections import ChainMap

import tensorflow as tf
from fastestimator.op import get_inputs_by_op, get_op_from_mode, verify_ops, write_outputs_by_key
from fastestimator.op.tensorop import ModelOp
from fastestimator.schedule import Scheduler
from fastestimator.util.util import NonContext, flatten_list, get_num_devices, to_list
from tensorflow.python.framework import ops as tfops


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
        self.distribute_strategy = 0

    def prepare(self):
        """This function constructs the model specified in model definition and create replica of model
         for distributed training across multiple devices if there are multiple GPU available.
        """
        mode_list = ["train", "eval"]
        all_output_keys = []
        all_models = []
        for mode in mode_list:
            signature_epoch, mode_ops = self._get_signature_epoch(mode)
            epoch_ops_map = {}
            epoch_model_map = {}
            for epoch in signature_epoch:
                epoch_ops = []
                epoch_model = []
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
                        if op.model not in all_models:
                            all_models.append(op.model)
                assert epoch_model, "Network has no model for epoch {}".format(epoch)
                epoch_ops_map[epoch] = epoch_ops
                epoch_model_map[epoch] = epoch_model
            self.op_schedule[mode] = Scheduler(epoch_dict=epoch_ops_map)
            self.model_schedule[mode] = Scheduler(epoch_dict=epoch_model_map)
        self.all_output_keys = set(flatten_list(all_output_keys)) - {None}
        for model in all_models:
            assert model.model_name not in self.model, "duplicated model name: {}".format(model.model_name)
            self.model[model.model_name] = model
            if self.distribute_strategy == 0:
                self.distribute_strategy = model.distribute_strategy

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
        model_list = self.model_schedule[mode].get_current_value(epoch)
        epoch_losses = []
        for model in model_list:
            if model.loss_name not in epoch_losses:
                epoch_losses.append(model.loss_name)
        self.epoch_losses = epoch_losses
        return ops, model_list, epoch_losses

    def run_step(self, batch, ops, model_list, epoch_losses, state, warm_up=False):
        """Function that calculates the loss and gradients for curent step in training. It also constructs the higher
        level computational graph between the models before the training.

        Args:
            batch : dictionary that contains batch data and predictions from last epoch
            ops : Model operation dictionary that contains 'Inputs','Mode', and 'Outputs'
            model_list : List of the models
            epoch_losses : List of epoch losses.
            state : run time dictionary that contains following keys 'mode' and 'batch size'
            warm_up (bool, optional): Specifies if it's in warm up phase or not. Defaults to False.

        Returns:
            dictionary containing the predictions of current epoch
        """
        prediction = {}
        batch = ChainMap(prediction, batch)
        mode = state["mode"]
        global_batch_size = state["batch_size"]
        num_model = len(model_list)
        # use gradient tape for train, otherwise use a dummy tape
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward(batch, state, ops)
            reduced_loss = self._reduce_loss(batch, global_batch_size, epoch_losses, warm_up)
        # update model only for train mode
        if mode == "train":
            for idx in range(num_model):
                model = model_list[idx]
                loss = reduced_loss[model.loss_name]
                optimizer = model.optimizer
                if warm_up:
                    with tfops.init_scope():  # pylint: disable=not-context-manager
                        _ = optimizer.iterations
                        optimizer._create_hypers()  # pylint: disable=protected-access
                        optimizer._create_slots(model_list[idx].trainable_variables)  # pylint: disable=protected-access
                else:
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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

    def _reduce_loss(self, batch, global_batch_size, epoch_losses, warm_up):
        reduced_loss = {}
        for loss_name in epoch_losses:
            element_wise_loss = batch[loss_name]
            if warm_up:
                assert element_wise_loss.ndim != 0 and element_wise_loss.shape[0] == global_batch_size / \
                    self.num_devices, "please make sure loss is element-wise loss"
            reduced_loss[loss_name] = tf.reduce_sum(element_wise_loss) / global_batch_size
        return reduced_loss


def build(model_def, model_name, optimizer, loss_name):
    """build keras model instance in FastEstimator

    Args:
        model_def (function): function definition of tf.keras model
        model_name (str, list, tuple): model name(s)
        optimizer (str, optimizer, list, tuple): optimizer(s)
        loss_name (str, list, tuple): loss name(s)

    Returns:
        model: model(s) compiled by FastEstimator
    """
    if get_num_devices() > 1:
        distribute_strategy = tf.distribute.MirroredStrategy()
    else:
        distribute_strategy = None
    with distribute_strategy.scope() if distribute_strategy else NonContext():
        model = to_list(model_def())
        model_name = to_list(model_name)
        optimizer = to_list(optimizer)
        loss_name = to_list(loss_name)
        assert len(model) == len(model_name) == len(optimizer) == len(loss_name)
        for idx, (m, m_n, o, l_n) in enumerate(zip(model, model_name, optimizer, loss_name)):
            model[idx] = _fe_compile(m, m_n, o, l_n, distribute_strategy)
    if len(model) == 1:
        model = model[0]
    return model


def _fe_compile(model, model_name, optimizer, loss_name, distribute_strategy):
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
    model.distribute_strategy = distribute_strategy
    model.fe_compiled = True
    return model
