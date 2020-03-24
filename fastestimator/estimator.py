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
import itertools
from collections import ChainMap, deque
from typing import Dict, Iterable, List, Optional, Set, Union

import tensorflow as tf
from tensorflow.python.distribute.input_lib import DistributedDataset
from torch.utils.data import DataLoader

from fastestimator.backend.to_shape import to_shape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.to_type import to_type
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.summary.system import System
from fastestimator.trace.io.best_model_saver import BestModelSaver
from fastestimator.trace.io.model_saver import ModelSaver
from fastestimator.trace.trace import EvalEssential, Logger, Trace, TrainEssential
from fastestimator.util.data import Data
from fastestimator.util.util import Suppressor, draw, to_list, to_set


class Estimator:
    """Estimator is the highest level class that user can directly use for traning a model (estimator.fit). It wraps
    up `Pipeline`, `Network`, `Trace` objects together and defines the whole optimization process with other training
    necessary information.
    Args:
        pipeline: Pipeline object that defines the data processing workflow. It should be an instance of
            `fastestimator.pipepline.pipeline.Pipeline`
        network: Network object that defines models and their external connection. It should be an instance of
            `fastestimator.network.network.Network`
        epochs: Number of epochs to run.
        max_steps_per_epoch: maximum steps to run for each epoch. If None, all data will be used
        traces: List of the traces objects to run during training. If None, there will be only basic
            traces.
        log_steps: Interval steps of logging. Defaults to 100.
        monitor_names: Additional keys to print in logger
    """
    pipeline: Pipeline
    traces: List[Trace]
    monitor_names: Set[str]
    trace_inputs: Dict[str, Set[str]]  # {mode: {keys}}
    trace_outputs: Dict[str, Set[str]]  # {mode: {keys}}

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
        self.traces = [trace for trace in to_list(traces)]
        assert log_steps is None or log_steps >= 0, \
            "log_steps must be None or positive (or 0 to disable only train logging)"
        self.monitor_names = to_set(monitor_names)
        self.system = System(network=network,
                             log_steps=log_steps,
                             total_epochs=epochs,
                             max_steps_per_epoch=max_steps_per_epoch)
        self.trace_inputs = dict()
        self.trace_outputs = dict()
        self._prepare_traces()
        self._check_keys()

    def fit(self, summary: Optional[str] = None):
        draw()
        self.system.reset(summary)
        self._warmup()
        self._start_train()
        return self.system.summary or None

    def test(self, summary: Optional[str] = None):
        self.system.reset_for_test(summary)
        self._start_test()
        return self.system.summary or None

    def _sort_traces(self):
        # This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
        # representation in order to get the slightly better asymptotic runtime complexity
        modes = self.pipeline.get_modes()
        pipeline_all_outputs = set(
            itertools.chain.from_iterable(
                [self.pipeline.get_all_output_keys(mode, self.system.total_epochs) for mode in modes]))
        network_all_outputs = set(
            itertools.chain.from_iterable(
                [self.network.get_all_output_keys(mode, self.system.total_epochs) for mode in modes]))
        sorted_traces = []
        available_outputs = set(self.system.__dict__.keys()) | pipeline_all_outputs | network_all_outputs
        end_traces = deque()
        intermediate_traces = deque()
        intermediate_outputs = set()
        trace_deque = deque(self.traces)
        while trace_deque:
            trace = trace_deque.popleft()
            ins = set(trace.inputs)
            outs = set(trace.outputs)
            if not ins:
                sorted_traces.append(trace)
                available_outputs |= outs
            elif "*" in ins:
                if outs:
                    end_traces.appendleft(trace)
                else:
                    end_traces.append(trace)
            elif ins <= available_outputs:
                sorted_traces.append(trace)
                available_outputs |= outs
            else:
                intermediate_traces.append(trace)
                intermediate_outputs |= outs

        already_seen = set()
        while intermediate_traces:
            trace = intermediate_traces.popleft()
            ins = set(trace.inputs)
            outs = set(trace.outputs)
            already_seen.add(trace)
            if ins <= available_outputs:
                sorted_traces.append(trace)
                available_outputs |= outs
                already_seen.clear()
            elif ins <= (available_outputs | intermediate_outputs):
                intermediate_traces.append(trace)
            else:
                raise AssertionError("Trace {} has unsatisfiable inputs: {}".format(
                    type(trace).__name__, ", ".join(ins - (available_outputs | intermediate_outputs))))

            if intermediate_traces and len(already_seen) == len(intermediate_traces):
                raise AssertionError("Dependency cycle detected amongst traces: {}".format(", ".join(
                    [type(tr).__name__ for tr in already_seen])))
        sorted_traces.extend(list(end_traces))
        self.traces = sorted_traces

    def _prepare_traces(self):
        modes = self.pipeline.get_modes()
        loss_keys = self.network.get_loss_keys()
        if "train" in modes:
            self.traces.insert(0, TrainEssential(loss_keys=loss_keys))
        if "eval" in modes:
            self.traces.insert(1, EvalEssential(loss_keys=loss_keys))
        if self.system.log_steps is not None:
            self.traces.append(Logger(extra_log_keys=self.monitor_names))
        for mode in modes:
            trace_inputs = set()
            # '*' is a reserved key for traces to indicate that they want to receive all available output
            trace_outputs = {"*"}
            for trace in self.traces:
                if not trace.mode or mode in trace.mode:
                    trace_inputs.update(trace.inputs)
                    trace_outputs.update(trace.outputs)
            self.trace_inputs[mode] = trace_inputs
            self.trace_outputs[mode] = trace_outputs
        no_save_warning = True
        for trace in self.traces:
            trace.system = self.system
            if isinstance(trace, (ModelSaver, BestModelSaver)):
                no_save_warning = False
        if no_save_warning:
            print("FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.")
        self._sort_traces()

    def _warmup(self):
        pipeline_signature_epochs = self.pipeline.get_signature_epochs(self.system.total_epochs)
        network_signature_epochs = self.network.get_signature_epochs(self.system.total_epochs)
        signature_epochs = pipeline_signature_epochs | network_signature_epochs
        for epoch in signature_epochs:
            for mode in self.pipeline.get_modes():
                loader = self._configure_loader(self.pipeline.get_loader(mode, epoch))
                self.network.load_epoch(mode, epoch, output_keys=self.trace_inputs[mode], warmup=True)
                with Suppressor():
                    if isinstance(loader, tf.data.Dataset):
                        batch = list(loader.take(1))[0]
                    else:
                        batch = next(iter(loader))
                batch = self._configure_tensor(loader, batch)
                self.network.run_step(batch)
                self.network.unload_epoch()

    def _check_keys(self):
        for mode in self.pipeline.get_modes():
            pipeline_all_outputs = self.pipeline.get_all_output_keys(mode, self.system.total_epochs)
            network_all_outputs = self.network.get_all_output_keys(mode, self.system.total_epochs)
            unmet_requirements = self.trace_inputs[mode] - (pipeline_all_outputs
                                                            | network_all_outputs | self.trace_outputs[mode])
            assert not unmet_requirements, "found missing key(s) during {}: {}".format(mode, unmet_requirements)

    def _start_train(self):
        self._run_traces_on_begin({"train", "eval"})
        try:
            for self.system.epoch_idx in range(1, self.system.total_epochs + 1):
                if "train" in self.pipeline.get_modes():
                    self.system.mode = "train"
                    self._run_epoch()
                if "eval" in self.pipeline.get_modes():
                    self.system.mode = "eval"
                    self._run_epoch()
        except EarlyStop:
            pass  # On early stopping we still want to run the final traces and return results
        self._run_traces_on_end({"train", "eval"})

    def _start_test(self):
        self._run_traces_on_begin({"test"})
        self._run_epoch()
        self._run_traces_on_end({"test"})

    def _run_epoch(self):
        self._run_traces_on_epoch_begin()
        loader = iter(self._configure_loader(self.pipeline.get_loader(self.system.mode, self.system.epoch_idx)))
        self.network.load_epoch(mode=self.system.mode,
                                epoch=self.system.epoch_idx,
                                output_keys=self.trace_inputs[self.system.mode])
        self.system.batch_idx = None
        while True:
            try:
                with Suppressor():
                    batch = next(loader)
                self.system.update_batch_idx()
                self._run_traces_on_batch_begin()
                batch = self._configure_tensor(loader, batch)
                batch, prediction = self.network.run_step(batch)
                self._run_traces_on_batch_end(batch, prediction)
                if self.system.batch_idx == self.system.max_steps_per_epoch and self.system.mode == "train":
                    break
            except StopIteration:
                break
        self.network.unload_epoch()
        self._run_traces_on_epoch_end()

    def _configure_loader(self, loader):
        new_loader = loader
        if isinstance(new_loader, DataLoader) and isinstance(self.network, TFNetwork):
            batch = to_tensor(loader.dataset[0], target_type="tf")
            data_type = to_type(batch)
            data_shape = to_shape(batch, add_batch=True, exact_shape=False)
            new_loader = tf.data.Dataset.from_generator(lambda: loader, data_type, output_shapes=data_shape)
            new_loader = new_loader.prefetch(1)
        if isinstance(new_loader, tf.data.Dataset):
            if self.system.max_steps_per_epoch and self.system.mode == "train":
                new_loader = new_loader.take(self.system.max_steps_per_epoch)
            if isinstance(tf.distribute.get_strategy(),
                          tf.distribute.MirroredStrategy) and not isinstance(new_loader, DistributedDataset):
                new_loader = tf.distribute.get_strategy().experimental_distribute_dataset(new_loader)
        return new_loader

    def _configure_tensor(self, loader, batch):
        if isinstance(loader, tf.data.Dataset) and isinstance(self.network, TorchNetwork):
            batch = to_tensor(batch, target_type="torch")
        return batch

    def _run_traces_on_begin(self, run_modes: Set[str]):
        data = Data()
        for trace in self.traces:
            if not trace.mode or trace.mode & run_modes:
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

    def _run_traces_on_end(self, run_modes: Set[str]):
        data = Data()
        for trace in self.traces:
            if not trace.mode or trace.mode & run_modes:
                trace.on_end(data)

    def _check_early_exit(self):
        if self.system.stop_training:
            raise EarlyStop


class EarlyStop(Exception):
    pass
