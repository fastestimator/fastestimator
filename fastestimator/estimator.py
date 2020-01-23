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

from fastestimator.backend import torch_to_tf
from fastestimator.network import Network
from fastestimator.op.op import get_inputs_by_key
from fastestimator.op.tensorop.model import UpdateOp
from fastestimator.pipeline import BasePipeline
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
        steps_per_epoch (int, optional): Number of steps to run for each epoch.
        traces (list, optional): List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps (int, optional): Interval steps of logging. Defaults to 100.
        monitor_names (str, list): Additional keys to print in logger
    """
    pipeline: BasePipeline
    epochs: int
    steps_per_epoch: Optional[int]
    traces: List[Trace]
    log_steps: int

    def __init__(self,
                 pipeline: BasePipeline,
                 network: Network,
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
        self.trace_inputs = set()
        self.do_eval = False
        self.system = None

    def fit(self):
        draw()
        self._prepare_estimator()
        self._prepare_network()
        self._prepare_pipeline()
        return self._start()

    def _prepare_pipeline(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.pipeline.get_num_steps(mode="train", epoch=0)
        self.system.total_steps = self.epochs * self.steps_per_epoch

    def _prepare_network(self):
        self.network.exported_keys = self.network.op_outputs.intersection(self.trace_inputs)

    def _prepare_estimator(self):
        self.do_eval = "eval" in self.pipeline.get_modes()  # TODO - Scheduling
        self._prepare_traces()
        self._prepare_system()

    def _prepare_system(self):
        self.system = System(mode="train",
                             global_step=0,
                             batch_size=None,
                             num_devices=get_num_devices(),
                             log_steps=self.log_steps,
                             total_epochs=self.epochs,
                             total_steps=None,
                             epoch_idx=0,
                             batch_idx=0)
        for trace in self.traces:
            trace.system = self.system

    def _prepare_traces(self):
        loss_keys = self._initialize_loss_keys()
        if self.traces is None:
            self.traces = []
        self.traces = to_list(self.traces)
        self.monitor_names = set(filter(None, to_list(self.monitor_names))).union(loss_keys)
        self.traces.insert(0, TrainEssential(monitor_names=self.monitor_names))
        if self.do_eval:
            self.traces.append(EvalEssential(loss_keys=loss_keys))
        for trace in self.traces:
            self.trace_inputs = self.trace_inputs.union(set(filter(None, to_list(trace.inputs))))
            self.monitor_names = self.monitor_names.union(set(filter(None, to_list(trace.log_names))))
        self.traces.append(Logger(log_names=self.monitor_names, loss_names=loss_keys))

    def _initialize_loss_keys(self) -> Set[str]:
        loss_keys = set()
        for op in self.network.ops:
            if isinstance(op, UpdateOp):
                loss_keys = loss_keys.union(set(to_list(op.inputs)))
        return loss_keys

    def _start(self):
        self._run_traces_on_begin()
        for self.system.epoch_idx in range(self.epochs):
            self.system.mode = "train"
            self._run_epoch()
            if self.do_eval:
                self.system.mode = "eval"
                self._run_epoch()
            self.system.update_epoch_idx()
        self._run_traces_on_end()

    def _run_epoch(self):
        self._run_traces_on_epoch_begin()
        self.system.batch_size = self.pipeline.get_batch_size(mode=self.system.mode, epoch=self.system.epoch_idx)
        ds_iter = self.pipeline.get_iterator(mode=self.system.mode, epoch=self.system.epoch_idx)
        for self.system.batch_idx, batch in enumerate(ds_iter):
            if self.network.framework == "tensorflow":
                batch = torch_to_tf(batch)  # TODO - this should maybe be handled somewhere else...
            if self.system.batch_idx == self.steps_per_epoch and self.system.mode == "train":
                break
            self._run_traces_on_batch_begin()
            prediction = self.network.run_step(batch, {"mode": self.system.mode, "epoch": self.system.epoch_idx})
            self._run_traces_on_batch_end(batch, prediction)
            if self.system.mode == "train":
                self.system.update_global_step()
        self._run_traces_on_epoch_end()

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
                 batch_size: int,
                 num_devices: int,
                 log_steps: int,
                 total_epochs: int,
                 total_steps: int,
                 epoch_idx: int,
                 batch_idx: int):
        self.mode = mode
        self.global_step = global_step
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.buffer = {}

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
