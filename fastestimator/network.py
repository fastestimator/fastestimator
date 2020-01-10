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
from typing import Union, List, Dict, Any, Set, Mapping

import tensorflow as tf
import torch

from fastestimator.op import get_inputs_by_op, get_ops_by_mode, write_outputs_by_key, TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.util.util import NonContext, to_list


class Network:
    """A class representing network operations for FastEstimator model training.
    Args:
        ops : Specifies the series of operations for training model
    """
    def __init__(self, ops: Union[TensorOp, List[TensorOp]]):
        self.ops = to_list(ops)
        self.framework = None
        self.device = None
        self.model_list = []
        self.exported_keys = set()
        self.op_outputs = set()
        self.op_inputs = set()
        self._initial_check()

    def _initial_check(self):
        self._check_model()
        self._check_ops()
        self._check_device()

    def _check_ops(self):
        for op in self.ops:
            self.op_outputs = self.op_outputs.union(set(filter(None, to_list(op.outputs))))
            self.op_inputs = self.op_inputs.union(set(filter(None, to_list(op.inputs))))

    def _check_device(self):
        if self.framework == "pytorch":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                for model in self.model_list:
                    model.to(self.device)

    def prepare(self):
        pass

    def run_step(self, batch: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ops in Network
        Args:
            batch : dictionary that contains batch data after the pipeline
            state : dictionary that contains meta data
        Returns:
            dictionary containing the predictions of current epoch
        """
        ops = get_ops_by_mode(self.ops, state["mode"])
        batch = self._get_network_inputs(batch)
        if self.framework == "tensorflow":
            prediction = self._forward_tensorflow(batch, state, ops, to_list(self.exported_keys))
        else:
            prediction = self._forward_pytorch(batch, state, ops, self.exported_keys)
        return prediction

    def _get_network_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # In order to prevent sending unwanted data to gpu, this function extracts pipeline data required by network
        new_batch = {}
        extracted_keys = set(batch.keys()).intersection(self.op_inputs)
        for key in extracted_keys:
            new_batch[key] = batch[key]
        return new_batch

    def _forward_pytorch(self,
                         batch: Dict[str, Any],
                         state: Dict[str, Any],
                         ops: List[TensorOp],
                         exported_keys: Set[str]) -> Dict[str, Any]:
        prediction = {}
        mode = state["mode"]
        state['tape'] = None
        if self.device.type == "cuda":
            for key, val in batch.items():
                batch[key] = val.to(self.device)
        with torch.no_grad() if mode != "train" else NonContext():
            self._forward(batch, state, ops)
        for key in exported_keys:
            value = batch[key]
            if self.device.type == "cuda":
                value = value.to("cpu")
            prediction[key] = value
        return prediction

    @tf.function
    def _forward_tensorflow(self,
                            batch: Dict[str, Any],
                            state: Dict[str, Any],
                            ops: List[TensorOp],
                            exported_keys: List[str]) -> Dict[str, Any]:
        batch = ChainMap({}, batch)
        prediction = {}
        mode = state["mode"]
        # use gradient tape for tensorflow train, otherwise use a dummy tape
        with tf.GradientTape(persistent=True) if mode == "train" else NonContext() as tape:
            state['tape'] = tape
            self._forward(batch, state, ops)
        del state['tape']
        del tape
        for key in exported_keys:
            prediction[key] = batch[key]
        return prediction

    def _check_model(self):
        frameworks = []
        for op in self.ops:
            if isinstance(op, (ModelOp, UpdateOp)):
                if isinstance(op.model, tf.keras.Model):
                    frameworks.append("tensorflow")
                elif isinstance(op.model, torch.nn.Module):
                    frameworks.append("pytorch")
                if op.model not in self.model_list:
                    self.model_list.append(op.model)
        assert len(set(frameworks)) == 1, "please make sure either tensorflow or torch model is used in network"
        self.framework = frameworks.pop()

    @staticmethod
    def _forward(batch: Mapping[str, Any], state: Dict[str, Any], ops: List[TensorOp]):
        data = None
        for op in ops:
            data = get_inputs_by_op(op, batch, data)
            data = op.forward(data, state)
            if op.outputs:
                write_outputs_by_key(batch, data, op.outputs)


def build(
    model: Union[tf.keras.Model, torch.nn.Module, List[tf.keras.Model], List[torch.nn.Module]],
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


def _fe_compile(
    model: Union[tf.keras.Model, torch.nn.Module],
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
