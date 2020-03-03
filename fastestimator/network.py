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
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Set, Union

import tensorflow as tf

import torch
from fastestimator.backend import load_model
from fastestimator.op import TensorOp, get_current_ops, get_inputs_by_op, write_outputs_by_op
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, RepeatScheduler, Scheduler
from fastestimator.util.util import NonContext, lcms, to_list


class BaseNetwork:
    def __init__(self, ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]):
        self.ops = to_list(ops)
        self.models = to_list(_collect_models(ops))
        self._verify_inputs()
        self.effective_inputs = dict()
        self.effective_outputs = dict()
        self.epoch_models = set()

    def _verify_inputs(self):
        for op in self.ops:
            if isinstance(op, Scheduler):
                for epoch_op in op.get_all_values():
                    assert isinstance(epoch_op, TensorOp), "unsupported op format, must provide TensorOp in Network"
            else:
                assert isinstance(op, TensorOp), "unsupported op format, must provide TensorOp in Network"

    def load_epoch(self, mode: str, epoch: int) -> List[TensorOp]:
        epoch_ops = get_current_ops(self.ops, mode, epoch)
        self.epoch_models = set(op.model for op in epoch_ops if isinstance(op, (UpdateOp, ModelOp)))
        for model in self.epoch_models:
            if hasattr(model, "optimizer") and model.optimizer is not None:
                if isinstance(model.optimizer, Scheduler):
                    model.current_optimizer = model.optimizer.get_current_value(epoch)
                else:
                    model.current_optimizer = model.optimizer
        return epoch_ops

    def unload_epoch(self):
        pass

    def get_loss_keys(self) -> Set[str]:
        loss_keys = set()
        for op in self.ops:
            if isinstance(op, Scheduler):
                for epoch_op in op.get_all_values():
                    if isinstance(epoch_op, UpdateOp):
                        loss_keys.update(epoch_op.inputs)
            else:
                if isinstance(op, UpdateOp):
                    loss_keys.update(op.inputs)
        return loss_keys

    def get_effective_input_keys(self, mode: str, total_epochs: int) -> Set[str]:
        input_keys = set()
        produced_keys = set()
        for epoch in self.get_signature_epochs(total_epochs):
            for op in get_current_ops(self.ops, mode, epoch):
                input_keys.update(set(key for key in op.inputs if key not in produced_keys))
                produced_keys.update(op.outputs)
        return input_keys

    def get_all_output_keys(self, mode: str, total_epochs: int) -> Set[str]:
        output_keys = set()
        for epoch in self.get_signature_epochs(total_epochs):
            for op in get_current_ops(self.ops, mode, epoch):
                output_keys.update(op.outputs)
        return output_keys

    def get_signature_epochs(self, total_epochs: int) -> Set[int]:
        signature_epochs = {0}
        epoch_keys = {0}
        repeat_cycles = {1}
        for x in self.ops + [model.optimizer for model in self.models]:
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
    def _forward_batch(batch: MutableMapping[str, Any], state: Dict[str, Any], ops: List[TensorOp]):
        data = None
        for op in ops:
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_op(op, batch, data)

    def get_effective_batch_input(self, batch: MutableMapping[str, Any], mode: str) -> Dict[str, Any]:
        new_batch = {}
        for key in self.effective_inputs[mode]:
            if key in batch:
                new_batch[key] = batch[key]
        return new_batch

    def forward_step_eager(self,
                           batch: Dict[str, Any],
                           state: Dict[str, Any],
                           ops: List[TensorOp],
                           effective_outputs: List[str]):
        raise NotImplementedError

    def forward_step_static(self,
                            batch: Dict[str, Any],
                            state: Dict[str, Any],
                            ops: List[TensorOp],
                            effective_outputs: List[str]):

        return self.forward_step_eager(batch, state, ops, effective_outputs)


def _collect_models(ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]) -> Set[Union[tf.keras.Model, torch.nn.Module]]:
    models = set()
    for op in ops:
        if isinstance(op, Scheduler):
            models_in_schedule = set(x.model for x in op.get_all_values() if isinstance(x, (ModelOp, UpdateOp)))
            models.update(models_in_schedule)
        elif isinstance(op, (ModelOp, UpdateOp)):
            models.add(op.model)
    return models


# noinspection PyPep8Naming
def Network(ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]) -> BaseNetwork:
    models = _collect_models(ops)
    assert models, "cannot find model in Network ops"
    framework = set()
    for model in models:
        if isinstance(model, tf.keras.Model):
            framework.add("tf")
        elif isinstance(model, torch.nn.Module):
            framework.add("torch")
        else:
            framework.add("unknown")
    assert len(framework) == 1, "please make sure either tensorflow or torch model is used in network"

    framework = framework.pop()
    if framework == "tf":
        network = TFNetwork(ops)
    elif framework == "torch":
        network = TorchNetwork(ops)
    else:
        raise ValueError("Unkown model type")
    return network


class TorchNetwork(BaseNetwork):
    def __init__(self, ops: Iterable[Union[TensorOp, Scheduler[TensorOp]]]):
        super().__init__(ops)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_epoch(self, mode: str, epoch: int) -> List[TensorOp]:
        epoch_ops = super().load_epoch(mode, epoch)
        if self.device.type == "cuda":
            for model in self.epoch_models:
                model.to(self.device)
        return epoch_ops

    def unload_epoch(self):
        if self.device.type == "cuda":
            for model in self.epoch_models:
                model.to("cpu")

    def forward_step_eager(self,
                           batch: Dict[str, Any],
                           state: Dict[str, Any],
                           ops: List[TensorOp],
                           effective_outputs: List[str]) -> Dict[str, Any]:
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
    def forward_step_eager(self,
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

    @tf.function
    def forward_step_static(self,
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


def build(model_fn: Callable,
          optimizer_fn: Union[str, Scheduler, Callable, List[str], List[Callable], List[Scheduler], None],
          weights_path: Union[str, None, List[Union[str, None]]] = None,
          model_names: Union[str, List[str], None] = None
          ) -> Union[tf.keras.Model, torch.nn.Module, List[tf.keras.Model], List[torch.nn.Module]]:
    """Build model instances and associate them with optimizers
    Args:
        model_fn: function that define model(s)
        optimizer_fn: optimizer string/definition or list of optimizer instances/strings. For example:
                    tensorflow user can do optimizer_fn = lambda: tf.optimizers.Adam(lr=0.1),
                    pytorch user can do  optimizer_fn = lambda x: torch.optim.Adam(params=x, lr=0.1)
        model_names: names of the model that will be used for logging purpose. If None, name will be assigned.
        weights_path: weights path to load from. Defaults None.
    Returns:
        models: model(s) built by FastEstimator
    """
    if not hasattr(build, "count"):
        build.count = 0

    def _generate_model_names(num_names):
        names = ["model" if i + build.count == 0 else "model{}".format(i + build.count) for i in range(num_names)]
        build.count += num_names
        return names

    models, optimizer_fn = to_list(model_fn()), to_list(optimizer_fn)
    # check framework
    if isinstance(models[0], tf.keras.Model):
        framework = "tf"
    elif isinstance(models[0], torch.nn.Module):
        framework = "torch"
    else:
        raise ValueError("unrecognized model format: {}".format(type(models[0])))
    # multi-gpu handling
    if framework == "tf" and torch.cuda.device_count() > 1 and not isinstance(tf.distribute.get_strategy(),
                                                                              tf.distribute.MirroredStrategy):
        tf.distribute.experimental_set_strategy(tf.distribute.MirroredStrategy())
        models = to_list(model_fn())
    # generate names
    if not model_names:
        model_names = _generate_model_names(len(models))
    model_names = to_list(model_names)
    # load weights
    if weights_path:
        weights_path = to_list(weights_path)
    else:
        weights_path = [None] * len(models)
    assert len(models) == len(optimizer_fn) == len(weights_path) == len(model_names), \
        "Found inconsistency in number of models, optimizers, model_names or weights"
    #create optimizer
    for idx, (model, optimizer_def, weight, name) in enumerate(zip(models, optimizer_fn, weights_path, model_names)):
        models[idx] = _fe_compile(model, optimizer_def, weight, name, framework)
    if len(models) == 1:
        models = models[0]
    return models


def _fe_compile(model: Union[tf.keras.Model, torch.nn.Module],
                optimizer_fn: Union[str, Scheduler, Callable],
                weight: Union[str, None],
                name: str,
                framework: str) -> Union[tf.keras.Model, torch.nn.Module]:

    if isinstance(optimizer_fn, EpochScheduler):
        for epoch, optimizer_def in optimizer_fn.epoch_dict.items():
            optimizer_fn.epoch_dict[epoch] = _build_optimizer(optimizer_def, model, framework)
    elif isinstance(optimizer_fn, RepeatScheduler):
        for idx, optimizer_def in enumerate(optimizer_fn.repeat_list):
            optimizer_fn.repeat_list[idx] = _build_optimizer(optimizer_def, model, framework)
    else:
        optimizer_fn = _build_optimizer(optimizer_fn, model, framework)
    if weight:
        load_model(model, weight)
    model.optimizer = optimizer_fn
    model.model_name = name
    model.fe_compiled = True
    return model


def _build_optimizer(optimizer_fn: Union[str, Callable], model: Union[tf.keras.Model, torch.nn.Module],
                     framework: str) -> Union[tf.optimizers.Optimizer, torch.optim.Optimizer]:
    if isinstance(optimizer_fn, str):
        optimizer_fn = _optimizer_fn_from_string(optimizer_fn, framework)
    optimizer = _optimizer_fn_to_optimizer(optimizer_fn, model, framework)
    return optimizer


def _optimizer_fn_from_string(name: str, framework: str) -> Callable:
    tf_optimizer_fn = {
        'adadelta': tf.optimizers.Adadelta,
        'adagrad': tf.optimizers.Adagrad,
        'adam': tf.optimizers.Adam,
        'adamax': tf.optimizers.Adamax,
        'rmsprop': tf.optimizers.RMSprop,
        'sgd': tf.optimizers.SGD
    }
    pytorch_optimizer_fn = {
        'adadelta': lambda x: torch.optim.Adadelta(params=x),
        'adagrad': lambda x: torch.optim.Adagrad(params=x),
        'adam': lambda x: torch.optim.Adam(params=x),
        'adamax': lambda x: torch.optim.Adamax(params=x),
        'rmsprop': lambda x: torch.optim.RMSprop(params=x),
        'sgd': lambda x: torch.optim.SGD(params=x, lr=0.01)
    }
    if framework == "tf":
        optimizer_fn = tf_optimizer_fn[name]
    else:
        optimizer_fn = pytorch_optimizer_fn[name]
    return optimizer_fn


def _optimizer_fn_to_optimizer(optimizer_fn: Callable, model: Union[tf.keras.Model, torch.nn.Module],
                               framework: str) -> Union[tf.optimizers.Optimizer, torch.optim.Optimizer]:
    optimizer = None
    if optimizer_fn:
        if framework == "tf":
            optimizer = optimizer_fn()
            assert isinstance(optimizer, tf.optimizers.Optimizer)
        else:
            optimizer = optimizer_fn(model.parameters())
            assert isinstance(optimizer, torch.optim.Optimizer)
            optimizer.zero_grad()
    return optimizer
