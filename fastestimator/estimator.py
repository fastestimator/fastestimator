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
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import tensorflow as tf
from fastestimator.backend.to_shape import to_shape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.to_type import to_type
from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.summary.system import System, Summary
from fastestimator.trace.io.best_model_saver import BestModelSaver
from fastestimator.trace.io.model_saver import ModelSaver
from fastestimator.trace.trace import EvalEssential, Logger, Trace, TrainEssential
from fastestimator.util.data import Data
from fastestimator.util.util import Suppressor, draw, to_list, to_set
from tensorflow.python.distribute.input_lib import DistributedDataset
from torch.utils.data import DataLoader


class Estimator:
    """One class to rule them all.

    Estimator is the highest level class within FastEstimator. It is the class which is invoked to actually train
    (estimator.fit) or test (estimator.test) models. It wraps `Pipeline`, `Network`, `Trace` objects together and
    defines the whole optimization process.

    Args:
        pipeline: An fe.Pipeline object that defines the data processing workflow.
        network: An fe.Network object that contains models and other training graph definitions.
        epochs: The number of epochs to run.
        max_steps_per_epoch: Maximum steps to run for each epoch. If None, all data will be used.
        traces: What Traces to run during training. If None, only the system's default Traces will be included.
        log_steps: Frequency (in steps) for printing log messages. 0 to disable all step-based printing (though epoch
            information will still print). None to completely disable printing.
        monitor_names: Additional keys from the data dictionary to be written into the logs.
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

    def fit(self, summary: Optional[str] = None) -> Optional[Summary]:
        """Train the network for the number of epochs specified by the estimator's constructor.

        Args:
            summary: A name for the experiment. If provided, the log history will be recorded in-memory and returned as
                a summary object at the end of training.

        Returns:
            A summary object containing the training history for this session iff a `summary` name was provided.
        """
        draw()
        self.system.reset(summary)
        self._warmup()
        self._start_train()
        return self.system.summary or None

    def test(self, summary: Optional[str] = None) -> Optional[Summary]:
        """Run the pipeline / network in test mode for one epoch.

        Args:
            summary: A name for the experiment. If provided, the log history will be recorded in-memory and returned as
                a summary object at the end of training. If None, the default value will be whatever `summary` name was
                most recently provided to this Estimator's .fit() or .test() methods.

        Returns:
            A summary object containing the training history for this session iff the `summary` name is not None (after
            considering the default behavior above).
        """
        self.system.reset_for_test(summary)
        self._start_test()
        return self.system.summary or None

    def _sort_traces(self) -> None:
        """Sort traces to attempt to resolve any dependency issues.

        This is essentially a topological sort, but it doesn't seem worthwhile to convert the data into a graph
        representation in order to get the slightly better asymptotic runtime complexity

        Raises:
            AssertionError: If Traces have circular dependencies or require input keys which are not available.
        """
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

    def _prepare_traces(self) -> None:
        """Prepare information about the traces for training.

        Add default traces into the trace list, determine which keys from the data dictionary must be retained and
        brought back from the GPU to CPU for use in traces. This also prints a warning if no model saver trace is
        detected.
        """
        modes = self.pipeline.get_modes()
        loss_keys = self.network.get_loss_keys()
        if "train" in modes:
            self.traces.insert(0, TrainEssential(loss_keys=loss_keys))
        if "eval" in modes and loss_keys.issubset(self.network.get_all_output_keys("eval", self.system.total_epochs)):
            self.traces.insert(1, EvalEssential(loss_keys=loss_keys, monitor_names=self.monitor_names))
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

    def _warmup(self) -> None:
        """Perform a test run of each pipeline and network signature epoch to make sure that training won't fail later.

        Traces are not included in the warmup since they are likely to contain state variables which could become
        corrupted by running extra steps.
        """
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

    def _check_keys(self) -> None:
        """Ensure that all keys required as inputs for traces actually exist in the data dictionary.

        Raises:
            AssertionError: If traces require inputs which are not available in the data dictionary.
        """
        for mode in self.pipeline.get_modes():
            pipeline_all_outputs = self.pipeline.get_all_output_keys(mode, self.system.total_epochs)
            network_all_outputs = self.network.get_all_output_keys(mode, self.system.total_epochs)
            unmet_requirements = self.trace_inputs[mode] - (pipeline_all_outputs
                                                            | network_all_outputs | self.trace_outputs[mode])
            assert not unmet_requirements, "found missing key(s) during {}: {}".format(mode, unmet_requirements)

    def _start_train(self) -> None:
        """The outer training loop.

        This method invokes the trace on_begin method, runs the necessary 'train' and 'eval' epochs, and then invokes
        the trace on_end method.
        """
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

    def _start_test(self) -> None:
        """The outer testing loop.

        This method invokes the trace on_begin method, runs a 'test' epoch, and then invokes the trace on_end method.
        """
        self._run_traces_on_begin({"test"})
        self._run_epoch()
        self._run_traces_on_end({"test"})

    def _run_epoch(self) -> None:
        """A method to perform an epoch of activity.

        This method requires that the current mode and epoch already be specified within the self.system object.
        """
        loader = self._configure_loader(self.pipeline.get_loader(self.system.mode, self.system.epoch_idx))
        iterator = iter(loader)
        self.network.load_epoch(mode=self.system.mode,
                                epoch=self.system.epoch_idx,
                                output_keys=self.trace_inputs[self.system.mode])
        self.system.batch_idx = None
        self._run_traces_on_epoch_begin()
        while True:
            try:
                with Suppressor():
                    batch = next(iterator)
                if self.system.mode == "train":
                    self.system.update_global_step()
                self.system.update_batch_idx()
                self._run_traces_on_batch_begin()
                batch = self._configure_tensor(loader, batch)
                batch, prediction = self.network.run_step(batch)
                self._run_traces_on_batch_end(batch, prediction)
                if self.system.batch_idx == self.system.max_steps_per_epoch and self.system.mode == "train":
                    break
            except StopIteration:
                break
        self._run_traces_on_epoch_end()
        self.network.unload_epoch()

    def _configure_loader(self, loader: Union[DataLoader, tf.data.Dataset]) -> Union[DataLoader, tf.data.Dataset]:
        """A method to configure a given dataloader for use with this Estimator's Network.

        This method will ensure that the `loader` returns the correct data type (tf.Tensor or torch.Tensor) depending on
         the requirements of the Network. It also handles issues with multi-gpu data sharding.

        Args:
            loader: A data loader to be modified.

        Returns:
            The potentially modified dataloader to be used for training.
        """
        new_loader = loader
        if isinstance(new_loader, DataLoader) and isinstance(self.network, TFNetwork):
            add_batch = True
            if hasattr(loader.dataset, "dataset") and isinstance(loader.dataset.dataset, BatchDataset):
                add_batch = False
            batch = to_tensor(loader.dataset[0], target_type="tf")
            data_type = to_type(batch)
            data_shape = to_shape(batch, add_batch=add_batch, exact_shape=False)
            new_loader = tf.data.Dataset.from_generator(lambda: loader, data_type, output_shapes=data_shape)
            new_loader = new_loader.prefetch(1)
        if isinstance(new_loader, tf.data.Dataset):
            if self.system.max_steps_per_epoch and self.system.mode == "train":
                new_loader = new_loader.take(self.system.max_steps_per_epoch)
            if isinstance(tf.distribute.get_strategy(),
                          tf.distribute.MirroredStrategy) and not isinstance(new_loader, DistributedDataset):
                new_loader = tf.distribute.get_strategy().experimental_distribute_dataset(new_loader)
        return new_loader

    def _configure_tensor(self, loader: Union[DataLoader, tf.data.Dataset], batch: Dict[str, Any]) -> Dict[str, Any]:
        """A function to convert a batch of tf.Tensors to torch.Tensors if required.

        Returns:
            Either the original `batch`, or the `batch` converted to torch.Tensors if required.
        """
        if isinstance(loader, tf.data.Dataset) and isinstance(self.network, TorchNetwork):
            batch = to_tensor(batch, target_type="torch")
        return batch

    def _run_traces_on_begin(self, run_modes: Set[str]) -> None:
        """Invoke the on_begin methods of all of the traces.

        Args:
            run_modes: Which types of traces to run. Traces with mode==None will always run, but if a trace is mode
                restricted then this argument will determine whether it should be run or not. This is required since
                on_begin should be invoked a total of one time during .fit() even though both 'train' and 'eval' loops
                will take place.
        """
        data = Data()
        for trace in self.traces:
            if not trace.mode or trace.mode & run_modes:
                trace.on_begin(data)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self) -> None:
        """Invoke the on_epoch_begin methods of all of the traces for the current mode.
        """
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_epoch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self) -> None:
        """Invoke the on_batch_begin methods of all of the traces for the current mode.
        """
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_batch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_end(self, batch: Dict[str, Any], prediction: Dict[str, Any]) -> None:
        """Invoke the on_batch_end methods of all of the traces for the current mode.

        Args:
            batch: The batch data which was provided by the pipeline.
            prediction: The prediction data which was generated by the network.
        """
        data = Data(ChainMap(prediction, batch))
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_batch_end(data)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self) -> None:
        """Invoke the on_epoch_end methods of all of the traces for the current mode.
        """
        data = Data()
        for trace in self.traces:
            if not trace.mode or self.system.mode in trace.mode:
                trace.on_epoch_end(data)
        self._check_early_exit()

    def _run_traces_on_end(self, run_modes: Set[str]) -> None:
        """Invoke the on_end methods of all of the traces.

        Args:
            run_modes: Which types of traces to run. Traces with mode==None will always run, but if a trace is mode
                restricted then this argument will determine whether it should be run or not. This is required since
                on_end should be invoked a total of one time during .fit() even though both 'train' and 'eval' loops
                will take place.
        """
        data = Data()
        for trace in self.traces:
            if not trace.mode or trace.mode & run_modes:
                trace.on_end(data)

    def _check_early_exit(self) -> None:
        """Determine whether training should be prematurely aborted.

        Raises:
            EarlyStop: If the system.stop_training flag has been set to True.
        """
        if self.system.stop_training:
            raise EarlyStop


class EarlyStop(Exception):
    """An exception raised when the system.stop_training flag is flipped by a Trace in order to abort the training.
    """
    pass
