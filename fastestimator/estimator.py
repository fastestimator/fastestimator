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
from typing import Any, Iterable, List, Optional, Set, Union

import tensorflow as tf

from fastestimator.backend import to_tensor
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.op.op import get_inputs_by_key
from fastestimator.op.tensorop.model import UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace import EvalEssential, Logger, Trace, TrainEssential
from fastestimator.util.util import draw, get_num_devices, to_list


class Estimator:
    """Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wraps
    up `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other training
    necessary information.
    Args:
        pipeline (obj): Pipeline object that defines the data processing workflow. It should be an instance of
            `fastestimator.pipepline.pipeline.Pipeline`
        network (obj): Network object that defines models and their external connection. It should be an instance of
            `fastestimator.network.network.Network`
        epochs (int): Number of epooch to run.
        steps_per_epoch (int, optional): maximum steps to run for each epoch. If None, all data will be used
        traces (list, optional): List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps (int, optional): Interval steps of logging. Defaults to 100.
        monitor_names (str, list): Additional keys to print in logger
    """
    pipeline: Pipeline
    epochs: int
    steps_per_epoch: Optional[int]
    traces: List[Trace]
    log_steps: int

    def __init__(self,
                 pipeline: Pipeline,
                 network: BaseNetwork,
                 epochs: int,
                 steps_per_epoch: Optional[int] = None,
                 traces: Union[Trace, Iterable[Trace]] = None,
                 log_steps: int = 100,
                 monitor_names: Optional[str] = None):
        self.pipeline = pipeline
        self.network = network
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.traces = [] if traces is None else to_list(traces)
        self.log_steps = log_steps
        assert log_steps is None or log_steps > 0, "log_steps must be positive or None"
        self.monitor_names = monitor_names
        self.system = None
        self.trace_inputs = dict()

    def fit(self):
        draw()
        self.traces = self._prepare_traces()
        self.system = self._prepare_system()
        self._check_keys()
        self._warmup()
        return self._start()

    def _warmup(self):
        pass

    def _check_keys(self):
        for mode in self.pipeline.get_modes():
            pipeline_all_outputs = self.pipeline.get_all_output_keys(mode, self.epochs)
            network_all_outputs = self.network.get_all_output_keys(mode, self.epochs)
            assert self.trace_inputs[mode].issubset(pipeline_all_outputs | network_all_outputs), "found missing key"
            self.network.effective_inputs[mode] = self.network.get_effective_input_keys(mode, self.epochs)
            self.network.effective_outputs[mode] = network_all_outputs.intersection(self.trace_inputs[mode])

    def _prepare_system(self):
        system = System(mode="train",
                        global_step=0,
                        num_devices=get_num_devices(),
                        log_steps=self.log_steps,
                        total_epochs=self.epochs,
                        epoch_idx=0,
                        batch_idx=0)
        system.network = self.network
        for trace in self.traces:
            trace.system = system
        return system

    def _prepare_traces(self):
        self.trace_inputs = dict()
        traces = [trace for trace in self.traces]
        loss_keys = self.network.get_loss_keys()
        monitor_names = set(filter(None, to_list(self.monitor_names))).union(loss_keys)
        traces.insert(0, TrainEssential(monitor_names=monitor_names))
        modes = self.pipeline.get_modes() - {"test"}
        if "eval" in modes:
            traces.append(EvalEssential(loss_keys=loss_keys))
        for mode in modes:
            trace_inputs = set()
            for trace in traces:
                if trace.mode is None or mode in to_list(trace.mode):
                    trace_inputs.update(filter(None, to_list(trace.inputs)))
                    monitor_names.update(filter(None, to_list(trace.log_names)))
            self.trace_inputs[mode] = trace_inputs
        traces.append(Logger(log_names=monitor_names, loss_names=loss_keys))
        return traces

    def _start(self):
        self._run_traces_on_begin()
        for self.system.epoch_idx in range(self.epochs):
            self.system.mode = "train"
            self._run_epoch()
            if "eval" in self.pipeline.get_modes():
                self.system.mode = "eval"
                self._run_epoch()
            self.system.update_epoch_idx()
        self._run_traces_on_end()

    def _run_epoch(self):
        self._run_traces_on_epoch_begin()
        self.network.load_epoch(mode=self.system.mode, epoch=self.system.epoch_idx)
        self.system.loader = self.pipeline.get_loader(mode=self.system.mode, epoch=self.system.epoch_idx)
        for self.system.batch_idx, batch in enumerate(self.system.loader):
            if self.system.batch_idx == self.steps_per_epoch and self.system.mode == "train":
                break
            batch = self._configure_tensor(batch)
            self._run_traces_on_batch_begin()
            prediction = self.network.run_step(batch, {"mode": self.system.mode})
            self._run_traces_on_batch_end(batch, prediction)
            if self.system.mode == "train":
                self.system.update_global_step()
        self._run_traces_on_epoch_end()

    def _configure_tensor(self, batch):
        if isinstance(self.system.loader, tf.data.Dataset):
            if isinstance(self.network, TorchNetwork):
                batch = to_tensor(batch, target_type="torch")
        elif isinstance(self.network, TFNetwork):
            batch = to_tensor(batch, target_type="tensorflow")
        return batch

    def _run_traces_on_begin(self):
        for trace in self.traces:
            trace.on_begin()
        self.system.clear_buffer()

    def _run_traces_on_epoch_begin(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_epoch_begin()
        self.system.clear_buffer()

    def _run_traces_on_batch_begin(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_batch_begin()
        self.system.clear_buffer()

    def _run_traces_on_batch_end(self, batch, prediction):
        batch = ChainMap(prediction, batch)
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                if trace.inputs:
                    data = get_inputs_by_key(batch, trace.inputs)
                else:
                    data = None
                trace.on_batch_end(data)
        self.system.clear_buffer()

    def _run_traces_on_epoch_end(self):
        for trace in self.traces:
            if trace.mode is None or self.system.mode in trace.mode:
                trace.on_epoch_end()
        self.system.clear_buffer()

    def _run_traces_on_end(self):
        for trace in self.traces:
            trace.on_end()
        self.system.clear_buffer()


class System:
    def __init__(self,
                 mode: str,
                 global_step: int,
                 num_devices: int,
                 log_steps: int,
                 total_epochs: int,
                 epoch_idx: int,
                 batch_idx: int):
        self.mode = mode
        self.global_step = global_step
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.buffer = {}
        self.loader = None
        self.network = None

    def add_buffer(self, key: str, value: Any):
        self.buffer[key] = value

    def clear_buffer(self):
        del self.buffer
        self.buffer = {}

    def read_buffer(self, key: str) -> Any:
        return self.buffer[key]

    def update_epoch_idx(self):
        self.epoch_idx += 1

    def update_global_step(self):
        self.global_step += 1
