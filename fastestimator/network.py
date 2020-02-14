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
from typing import Any, Dict, List, Mapping, Set, Union, Iterable

import tensorflow as tf
import torch
from fastestimator.op import TensorOp, get_current_ops, get_inputs_by_op, write_outputs_by_key
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, RepeatScheduler, Scheduler
from fastestimator.util.util import NonContext, lcms, to_list


class BaseNetwork:
    def __init__(self, ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]):
        self.ops = to_list(ops)
        self._verify_inputs()
        self.effective_inputs = dict()
        self.effective_outputs = dict()
        self.epoch_ops = []

    def _verify_inputs(self):
        for op in self.ops:
            if isinstance(op, Scheduler):
                for epoch_op in op.get_all_values():
                    assert isinstance(epoch_op, TensorOp), "unsupported op format, must provide TensorOp in Network"
            else:
                assert isinstance(op, TensorOp), "unsupported op format, must provide TensorOp in Network"

    def load_epoch(self, mode: str, epoch: int):
        self.epoch_ops = get_current_ops(self.ops, mode, epoch)

    def unload_epoch(self):
        pass

    def get_loss_keys(self) -> Set[str]:
        loss_keys = set()
        for op in self.ops:
            if isinstance(op, Scheduler):
                for epoch_op in op.get_all_values():
                    if isinstance(epoch_op, UpdateOp):
                        loss_keys.update(to_list(epoch_op.inputs))
            else:
                if isinstance(op, UpdateOp):
                    loss_keys.update(to_list(op.inputs))
        return loss_keys

    def get_effective_input_keys(self, mode: str, total_epochs: int) -> Set[str]:
        input_keys = set()
        produced_keys = set()
        for epoch in self.get_signature_epochs(total_epochs):
            for op in get_current_ops(self.ops, mode, epoch):
                input_keys.update(set(key for key in to_list(op.inputs) if key not in produced_keys))
                produced_keys.update(to_list(op.outputs))
        return input_keys

    def get_all_output_keys(self, mode: str, total_epochs: int) -> Set[str]:
        output_keys = set()
        for epoch in self.get_signature_epochs(total_epochs):
            for op in get_current_ops(self.ops, mode, epoch):
                output_keys.update(to_list(op.outputs))
        return output_keys

    def get_signature_epochs(self, total_epochs: int):
        signature_epochs = {0}
        epoch_keys = {0}
        repeat_cycles = {1}
        for x in self.ops:
            if isinstance(x, EpochScheduler):
                epoch_keys.update(x.epoch_dict.keys())
            elif isinstance(x, RepeatScheduler):
                repeat_cycles.add(x.cycle_length)
        least_common_cycle = lcms(*repeat_cycles)
        epoch_keys = sorted(epoch_keys)
        for idx, epoch in enumerate(epoch_keys):
            if idx + 1 < len(epoch_keys):
                signature_epochs.update(range(epoch, epoch + min(epoch_keys[idx + 1] - epoch, least_common_cycle)))
            else:
                signature_epochs.update(range(epoch, epoch + least_common_cycle))
        signature_epochs = set(epoch for epoch in signature_epochs if epoch < total_epochs)
        return signature_epochs

    @staticmethod
    def _forward_batch(batch: Mapping[str, Any], state: Dict[str, Any], ops: List[TensorOp]):
        data = None
        for op in ops:
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_key(batch, data, op.outputs)

    def run_step(self, batch: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# noinspection PyPep8Naming
def Network(ops):
    models = set()
    for op in ops:
        if isinstance(op, Scheduler):
            models_in_schedule = set(x.model for x in op.get_all_values() if isinstance(x, (ModelOp, UpdateOp)))
            models.update(models_in_schedule)
        elif isinstance(op, (ModelOp, UpdateOp)):
            models.add(op.model)
    assert models, "cannot find model in Network ops"

    framework = set()
    for model in models:
        if isinstance(model, tf.keras.Model):
            framework.add("tensorflow")
        elif isinstance(model, torch.nn.Module):
            framework.add("pytorch")
        else:
            framework.add("unknown")
    assert len(framework) == 1, "please make sure either tensorflow or torch model is used in network"

    framework = framework.pop()
    if framework == "tensorflow":
        network = TFNetwork(ops)
    elif framework == "pytorch":
        network = TorchNetwork(ops)
    else:
        raise ValueError("Unkown model type")
    return network


class TorchNetwork(BaseNetwork):
    def __init__(self, ops):
        super().__init__(ops)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch_models = set()

    def load_epoch(self, mode: str, epoch: int):
        self.epoch_ops = get_current_ops(self.ops, mode, epoch)
        if self.device.type == "cuda":
            self.epoch_models = set(op.model for op in self.epoch_ops if isinstance(op, (UpdateOp, ModelOp)))
            for model in self.epoch_models:
                model.to(self.device)

    def unload_epoch(self):
        if self.device.type == "cuda":
            for model in self.epoch_models:
                model.to("cpu")

    def run_step(self, batch: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ops in Network
        Args:
            batch : dictionary that contains batch data after the pipeline
            state : dictionary that contains meta data
        Returns:
            dictionary containing the predictions of current epoch
        """
        new_batch = {}
        mode = state["mode"]
        for key in self.effective_inputs[mode]:
            if key in batch:
                new_batch[key] = batch[key]
        prediction = self._forward_step(new_batch, state, self.epoch_ops, self.effective_outputs[mode])
        return prediction

    def _forward_step(self,
                      batch: Dict[str, Any],
                      state: Dict[str, Any],
                      ops: List[TensorOp],
                      effective_outputs: Set[str]) -> Dict[str, Any]:
        prediction = {}
        mode = state["mode"]
        state["tape"] = NonContext()
        if self.device.type == "cuda":
            for key, val in batch.items():
                batch[key] = val.to(self.device)
        with torch.no_grad() if mode != "train" else NonContext():
            self._forward_batch(batch, state, ops)
        for key in effective_outputs:
            if key in batch:
                value = batch[key]
                if self.device.type == "cuda":
                    value = value.to("cpu")
                prediction[key] = value
        return prediction


class TFNetwork(BaseNetwork):
    def __init__(self, ops):
        super().__init__(ops)

    def run_step(self, batch: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ops in Network
        Args:
            batch : dictionary that contains batch data after the pipeline
            state : dictionary that contains meta data
        Returns:
            dictionary containing the predictions of current epoch
        """
        new_batch = {}
        mode = state["mode"]
        for key in self.effective_inputs[mode]:
            if key in batch:
                new_batch[key] = batch[key]
        if state["warmup"]:
            prediction = self._forward_step_eager(new_batch, state, self.epoch_ops, self.effective_outputs[mode])
        else:
            prediction = self._forward_step_static(new_batch,
                                                   state,
                                                   self.epoch_ops,
                                                   to_list(self.effective_outputs[mode]))
        return prediction

    def _forward_step_eager(self,
                            batch: Dict[str, Any],
                            state: Dict[str, Any],
                            ops: List[TensorOp],
                            effective_outputs: Set[str]) -> Dict[str, Any]:
        batch = ChainMap({}, batch)
        prediction = {}
        mode = state["mode"]
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward_batch(batch, state, ops)
        del state['tape']
        del tape
        for key in effective_outputs:
            if key in batch:
                prediction[key] = batch[key]
        return prediction

    @tf.function
    def _forward_step_static(self,
                             batch: Dict[str, Any],
                             state: Dict[str, Any],
                             ops: List[TensorOp],
                             effective_outputs: List[str]) -> Dict[str, Any]:
        batch = ChainMap({}, batch)
        prediction = {}
        mode = state["mode"]
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward_batch(batch, state, ops)
        del state['tape']
        del tape
        for key in effective_outputs:
            if key in batch:
                prediction[key] = batch[key]
        return prediction


def build(model: Union[tf.keras.Model, torch.nn.Module, List[tf.keras.Model], List[torch.nn.Module]],
          optimizer: Union[str,
                           List[str],
                           tf.optimizers.Optimizer,
                           List[tf.optimizers.Optimizer],
                           torch.optim.Optimizer,
                           List[torch.optim.Optimizer]]
          ) -> Union[tf.keras.Model, torch.nn.Module, List[tf.keras.Model], List[torch.nn.Module]]:
    """Associate model instance(s) with optimizer(s)
    Args:
        model: model instances or list of model instances
        optimizer: optimizer instance/string or list of optimizer instance/string
    Returns:
        models: model(s) compiled by FastEstimator
    """
    models = to_list(model)
    optimizers = to_list(optimizer)
    assert len(models) == len(optimizers)
    for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        models[idx] = _fe_compile(model, optimizer)
    if len(models) == 1:
        models = models[0]
    return models


def _fe_compile(model: Union[tf.keras.Model, torch.nn.Module],
                optimizer: Union[str, tf.optimizers.Optimizer, torch.optim.Optimizer]
                ) -> Union[tf.keras.Model, torch.nn.Module]:
    # model instance check
    if isinstance(model, tf.keras.Model):
        framework = "tensorflow"
    elif isinstance(model, torch.nn.Module):
        framework = "pytorch"
    else:
        raise ValueError("unrecognized model format: {}".format(type(model)))

    # optimizer auto complete
    if isinstance(optimizer, str):
        tf_optimizer_fn = {
            'adadelta': tf.optimizers.Adadelta,
            'adagrad': tf.optimizers.Adagrad,
            'adam': tf.optimizers.Adam,
            'adamax': tf.optimizers.Adamax,
            'rmsprop': tf.optimizers.RMSprop,
            'sgd': tf.optimizers.SGD
        }
        pytorch_optimizer_fn = {
            'adadelta': torch.optim.Adadelta,
            'adagrad': torch.optim.Adagrad,
            'adam': torch.optim.Adam,
            'adamax': torch.optim.Adamax,
            'rmsprop': torch.optim.RMSprop,
            'sgd': torch.optim.SGD
        }
        if framework == "tensorflow":
            optimizer = tf_optimizer_fn[optimizer]()
        else:
            optimizer = pytorch_optimizer_fn[optimizer](params=model.parameters())

    # optimizer instance check
    if framework == "tensorflow":
        assert isinstance(optimizer, tf.optimizers.Optimizer)
    else:
        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizer.zero_grad()

    model.optimizer = optimizer
    model.fe_compiled = True
    return model
