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
from typing import Iterable, List, Optional, Set, Union

import tensorflow as tf
from torch.utils.data import DataLoader

from fastestimator.backend import to_tensor, to_type
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.trace import EvalEssential, Logger, Trace, TrainEssential
from fastestimator.util import Data, System
from fastestimator.util.util import draw, to_list, to_set


class Estimator:
    """Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wraps
    up `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other training
    necessary information.
    Args:
        pipeline: Pipeline object that defines the data processing workflow. It should be an instance of
            `fastestimator.pipepline.pipeline.Pipeline`
        network: Network object that defines models and their external connection. It should be an instance of
            `fastestimator.network.network.Network`
        epochs: Number of epooch to run.
        max_steps_per_epoch: maximum steps to run for each epoch. If None, all data will be used
        traces: List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps: Interval steps of logging. Defaults to 100.
        monitor_names: Additional keys to print in logger
    """
    pipeline: Pipeline
    traces: List[Trace]
    monitor_names: Set[str]

    def __init__(self,
                 pipeline: Pipeline,
                 network: BaseNetwork,
                 epochs: int,
                 max_steps_per_epoch: Optional[int] = None,
                 traces: Union[None, Trace, Iterable[Trace]] = None,
                 log_steps: Optional[int] = 100,
                 monitor_names: Union[None, str, Iterable[str]] = None):
        self.pipeline = pipeline
        self.network = network
        self.traces = to_list(traces)
        assert log_steps is None or log_steps >= 0, \
            "log_steps must be None or positive (or 0 to disable only train logging)"
        self.monitor_names = to_set(monitor_names)
        self.system = System(log_steps=log_steps, epochs=epochs, max_steps_per_epoch=max_steps_per_epoch)
        self.trace_inputs = dict()

    def fit(self):
        draw()
        self.traces = self._prepare_traces()
        self._prepare_system()
        self._check_keys()
        self._warmup()
        return self._start()

    def _warmup(self):
        pipeline_signature_epochs = self.pipeline.get_signature_epochs(self.system.epochs)
        network_signature_epochs = self.network.get_signature_epochs(self.system.epochs)
        signature_epochs = pipeline_signature_epochs | network_signature_epochs
        for epoch in signature_epochs:
            for mode in self.pipeline.get_modes():
                self.network.load_epoch(mode, epoch)
                loader = self._configure_loader(self.pipeline.get_loader(mode, epoch))
                if isinstance(loader, tf.data.Dataset):
                    batch = list(loader.take(1))[0]
                else:
                    batch = next(iter(loader))
                batch = self._configure_tensor(loader, batch)
                prediction = self.network.run_step(batch, {"mode": mode, "warmup": True})
                self.network.unload_epoch()

    def _check_keys(self):
        for mode in self.pipeline.get_modes():
            pipeline_all_outputs = self.pipeline.get_all_output_keys(mode, self.system.epochs)
            network_all_outputs = self.network.get_all_output_keys(mode, self.system.epochs)
            assert self.trace_inputs[mode].issubset(pipeline_all_outputs | network_all_outputs), "found missing key"
            self.network.effective_inputs[mode] = self.network.get_effective_input_keys(mode, self.system.epochs)
            self.network.effective_outputs[mode] = network_all_outputs.intersection(self.trace_inputs[mode])

    def _prepare_system(self):
        self.system.reset()
        self.system.network = self.network
        for trace in self.traces:
            trace.system = self.system

    def _prepare_traces(self):
        self.trace_inputs = dict()
        traces = [trace for trace in self.traces]
        loss_keys = self.network.get_loss_keys()
        traces.insert(0, TrainEssential())
        modes = self.pipeline.get_modes() - {"test"}
        if "eval" in modes:
            traces.insert(1, EvalEssential(loss_keys=loss_keys))
        if self.system.log_steps is not None:
            traces.append(Logger(extra_log_keys=self.monitor_names, loss_names=loss_keys))
        for mode in modes:
            trace_inputs = set()
            for trace in traces:
                if not trace.mode or mode in trace.mode:
                    trace_inputs.update(trace.inputs)
            self.trace_inputs[mode] = trace_inputs
        return traces

    def _start(self):
        try:
            self._run_traces_on_begin()
            for self.system.epoch_idx in range(self.system.epochs):
                self.system.mode = "train"
                self._run_epoch()
                if "eval" in self.pipeline.get_modes():
                    self.system.mode = "eval"
                    self._run_epoch()
        except EarlyStop:
            pass  # On early stopping we still want to run the final traces and return results
        self._run_traces_on_end()

    def _run_epoch(self):
        self._run_traces_on_epoch_begin()
        self.network.load_epoch(mode=self.system.mode, epoch=self.system.epoch_idx)
        loader = self._configure_loader(self.pipeline.get_loader(mode=self.system.mode, epoch=self.system.epoch_idx))
        for self.system.batch_idx, batch in enumerate(loader):
            if self.system.batch_idx == self.system.max_steps_per_epoch and self.system.mode == "train":
                break
            batch = self._configure_tensor(loader, batch)
            self._run_traces_on_batch_begin()
            prediction = self.network.run_step(batch, {"mode": self.system.mode, "warmup": False})
            self._run_traces_on_batch_end(batch, prediction)
            if self.system.mode == "train":
                self.system.update_global_step()
        self.network.unload_epoch()
        self._run_traces_on_epoch_end()

    def _configure_loader(self, loader):
        new_loader = loader
        if isinstance(loader, DataLoader) and isinstance(self.network, TFNetwork):
            data_type = to_type(to_tensor(loader.dataset[0], target_type="tensorflow"))
            new_loader = tf.data.Dataset.from_generator(lambda: loader, data_type)
            new_loader = new_loader.prefetch(1)
            if self.system.max_steps_per_epoch and self.system.mode == "train":
                new_loader = new_loader.take(self.system.max_steps_per_epoch)
        return new_loader

    def _configure_tensor(self, loader, batch):
        if isinstance(loader, tf.data.Dataset) and isinstance(self.network, TorchNetwork):
            batch = to_tensor(batch, target_type="torch")
        return batch

    def _run_traces_on_begin(self):
        data = Data()
        for trace in self.traces:
            trace.on_begin(data)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self):
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_epoch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self):
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_batch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_end(self, batch, prediction):
        data = Data(ChainMap(prediction, batch))
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_batch_end(data)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self):
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_epoch_end(data)
        self._check_early_exit()

    def _run_traces_on_end(self):
        data = Data()
        for trace in self.traces:
            trace.on_end(data)

    def _check_early_exit(self):
        if self.system.stop_training:
            raise EarlyStop


class EarlyStop(Exception):
    pass
