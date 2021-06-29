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
import os
import random
from collections import ChainMap
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.distribute.input_lib import DistributedDataset
from torch.utils.data import DataLoader

import fastestimator as fe
from fastestimator.backend.to_shape import to_shape
from fastestimator.backend.to_tensor import to_tensor
from fastestimator.backend.to_type import to_type
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler, get_current_items, get_signature_epochs
from fastestimator.summary.system import Summary, System
from fastestimator.trace.io.best_model_saver import BestModelSaver
from fastestimator.trace.io.model_saver import ModelSaver
from fastestimator.trace.io.restore_wizard import RestoreWizard
from fastestimator.trace.io.traceability import Traceability
from fastestimator.trace.trace import EvalEssential, Logger, TestEssential, Trace, TrainEssential, sort_traces
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import Suppressor, draw, to_list, to_set


@traceable()
class Estimator:
    """One class to rule them all.

    Estimator is the highest level class within FastEstimator. It is the class which is invoked to actually train
    (estimator.fit) or test (estimator.test) models. It wraps `Pipeline`, `Network`, `Trace` objects together and
    defines the whole optimization process.

    Args:
        pipeline: An fe.Pipeline object that defines the data processing workflow.
        network: An fe.Network object that contains models and other training graph definitions.
        epochs: The number of epochs to run.
        max_train_steps_per_epoch: Training will complete after n steps even if loader is not yet exhausted. If None,
            all data will be used.
        max_eval_steps_per_epoch: Evaluation will complete after n steps even if loader is not yet exhausted. If None,
            all data will be used.
        traces: What Traces to run during training. If None, only the system's default Traces will be included.
        log_steps: Frequency (in steps) for printing log messages. 0 to disable all step-based printing (though epoch
            information will still print). None to completely disable printing.
        monitor_names: Additional keys from the data dictionary to be written into the logs.
    """
    monitor_names: Set[str]
    traces_in_use: List[Union[Trace, Scheduler[Trace]]]
    system: System

    def __init__(self,
                 pipeline: Pipeline,
                 network: BaseNetwork,
                 epochs: int,
                 max_train_steps_per_epoch: Optional[int] = None,
                 max_eval_steps_per_epoch: Optional[int] = None,
                 traces: Union[None, Trace, Scheduler[Trace], Iterable[Union[Trace, Scheduler[Trace]]]] = None,
                 log_steps: Optional[int] = 100,
                 monitor_names: Union[None, str, Iterable[str]] = None):
        self.traces_in_use = []
        assert log_steps is None or log_steps >= 0, \
            "log_steps must be None or positive (or 0 to disable only train logging)"
        self.monitor_names = to_set(monitor_names) | network.get_loss_keys()
        self.system = System(network=network,
                             pipeline=pipeline,
                             traces=to_list(traces),
                             log_steps=log_steps,
                             total_epochs=epochs,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch,
                             system_config=self.fe_summary())

    @property
    def pipeline(self) -> Pipeline:
        return self.system.pipeline

    @property
    def network(self) -> BaseNetwork:
        return self.system.network

    @property
    def traces(self) -> List[Union[Trace, Scheduler[Trace]]]:
        return self.system.traces

    def fit(self, summary: Optional[str] = None, warmup: bool = True, eager: bool = False) -> Optional[Summary]:
        """Train the network for the number of epochs specified by the estimator's constructor.

        Args:
            summary: A name for the experiment. If provided, the log history will be recorded in-memory and returned as
                a summary object at the end of training.
            warmup: Whether to perform warmup before training begins. The warmup procedure will test one step at every
                epoch where schedulers cause the execution graph to change. This can take some time up front, but can
                also save significant heartache on epoch 300 when the training unexpectedly fails due to a tensor size
                mismatch.
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.

        Returns:
            A summary object containing the training history for this session iff a `summary` name was provided.
        """
        draw()
        self.system.reset(summary, self.fe_summary())
        self._prepare_traces(run_modes={"train", "eval"})
        if warmup:
            self._warmup(eager=eager)
        self._start(run_modes={"train", "eval"}, eager=eager)
        return self.system.summary or None

    def _prepare_traces(self, run_modes: Set[str]) -> None:
        """Prepare information about the traces for training.

        Add default traces into the traces_in_use list, also prints a warning if no model saver trace is detected.

        Args:
            run_modes: The current execution modes.
        """
        self.traces_in_use = [trace for trace in self.traces]
        if self.system.log_steps is not None:
            self.traces_in_use.append(Logger())
        # Look for any monitor names which should be automagically added.
        trace_outputs = set()
        extra_monitor_keys = set()
        for trace in sort_traces(get_current_items(self.traces_in_use, run_modes=run_modes)):
            trace_outputs.update(trace.outputs)
            extra_monitor_keys.update(trace.fe_monitor_names - trace_outputs)
        # Add the essential traces
        if "train" in run_modes:
            self.traces_in_use.insert(0, TrainEssential(monitor_names=self.monitor_names.union(extra_monitor_keys)))
            no_save_warning = True
            for trace in get_current_items(self.traces_in_use, run_modes=run_modes):
                if isinstance(trace, (ModelSaver, BestModelSaver)):
                    no_save_warning = False
            if no_save_warning:
                print("FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.")
        if "eval" in run_modes and "eval" in self.pipeline.get_modes():
            self.traces_in_use.insert(1, EvalEssential(monitor_names=self.monitor_names.union(extra_monitor_keys)))
        if "test" in run_modes and "test" in self.pipeline.get_modes():
            self.traces_in_use.insert(0, TestEssential(monitor_names=self.monitor_names.union(extra_monitor_keys)))
        # insert system instance to trace
        for trace in get_current_items(self.traces_in_use, run_modes=run_modes):
            trace.system = self.system

    def test(self, summary: Optional[str] = None, eager: bool = False) -> Optional[Summary]:
        """Run the pipeline / network in test mode for one epoch.

        Args:
            summary: A name for the experiment. If provided, the log history will be recorded in-memory and returned as
                a summary object at the end of training. If None, the default value will be whatever `summary` name was
                most recently provided to this Estimator's .fit() or .test() methods.
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.

        Returns:
            A summary object containing the training history for this session iff the `summary` name is not None (after
            considering the default behavior above).
        """
        self.system.reset_for_test(summary)
        self._prepare_traces(run_modes={"test"})
        self._start(run_modes={"test"}, eager=eager)
        return self.system.summary or None

    def _warmup(self, eager: bool = True) -> None:
        """Perform a test run of each pipeline and network signature epoch to make sure that training won't fail later.

        Traces are not executed in the warmup since they are likely to contain state variables which could become
        corrupted by running extra steps.

        Args:
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        all_traces = get_current_items(self.traces_in_use, run_modes={"train", "eval"})
        sort_traces(all_traces)  # This ensures that the traces can sort properly for on_begin and on_end
        monitor_names = self.monitor_names
        for mode in self.pipeline.get_modes() - {"test"}:
            scheduled_items = self.pipeline.get_scheduled_items(mode) + self.network.get_scheduled_items(
                mode) + self.get_scheduled_items(mode)
            signature_epochs = get_signature_epochs(scheduled_items, self.system.total_epochs, mode=mode)
            epochs_with_data = self.pipeline.get_epochs_with_data(total_epochs=self.system.total_epochs, mode=mode)
            for epoch in signature_epochs:
                if epoch not in epochs_with_data:
                    continue
                network_output_keys = self.network.get_all_output_keys(mode, epoch)
                network_input_keys = self.network.get_effective_input_keys(mode, epoch)
                trace_input_keys = set()
                trace_output_keys = {"*"}
                traces = get_current_items(self.traces_in_use, run_modes=mode, epoch=epoch)
                for idx, trace in enumerate(traces):
                    if idx > 0:  # ignore TrainEssential and EvalEssential's inputs for unmet requirement checking
                        trace_input_keys.update(trace.inputs)
                    trace_output_keys.update(trace.outputs)
                # key checking
                loader = self._configure_loader(
                    self.pipeline.get_loader(mode,
                                             epoch,
                                             output_keys=trace_input_keys - network_output_keys | network_input_keys))
                with Suppressor():
                    if isinstance(loader, tf.data.Dataset):
                        batch = list(loader.take(1))[0]
                    else:
                        batch = next(iter(loader))
                batch = self._configure_tensor(loader, batch)
                assert isinstance(batch, dict), "please make sure data output format is dictionary"
                pipeline_output_keys = to_set(batch.keys())

                monitor_names = monitor_names - (pipeline_output_keys | network_output_keys)
                unmet_requirements = trace_input_keys - (pipeline_output_keys | network_output_keys | trace_output_keys)
                assert not unmet_requirements, \
                    "found missing key(s) during epoch {} mode {}: {}".format(epoch, mode, unmet_requirements)
                sort_traces(traces, available_outputs=pipeline_output_keys | network_output_keys)
                trace_input_keys.update(traces[0].inputs)
                self.network.load_epoch(mode, epoch, output_keys=trace_input_keys, warmup=True, eager=eager)
                self.network.run_step(batch)
                self.network.unload_epoch()
        assert not monitor_names, "found missing key(s): {}".format(monitor_names)

    def get_scheduled_items(self, mode: str) -> List[Any]:
        """Get a list of items considered for scheduling.

        Args:
            mode: Current execution mode.

        Returns:
            List of schedulable items in estimator.
        """
        return self.traces_in_use

    def _start(self, run_modes: Set[str], eager: bool) -> None:
        """The outer training loop.

        This method invokes the trace on_begin method, runs the necessary 'train' and 'eval' epochs, and then invokes
        the trace on_end method.

        Args:
            run_modes: The current execution modes.
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        all_traces = sort_traces(get_current_items(self.traces_in_use, run_modes=run_modes))
        try:
            self._run_traces_on_begin(traces=all_traces)
            if "train" in run_modes or "eval" in run_modes:
                # If the training is re-starting from a restore wizard, it should re-run the last eval epoch
                if self.system.epoch_idx > 0 and "eval" in self.pipeline.get_modes(epoch=self.system.epoch_idx):
                    self.system.mode = "eval"
                    self._run_epoch(eager=eager)
                for self.system.epoch_idx in range(self.system.epoch_idx + 1, self.system.total_epochs + 1):
                    if "train" in self.pipeline.get_modes(epoch=self.system.epoch_idx):
                        self.system.mode = "train"
                        self._run_epoch(eager=eager)
                    if "eval" in self.pipeline.get_modes(epoch=self.system.epoch_idx):
                        self.system.mode = "eval"
                        self._run_epoch(eager=eager)
            else:
                self._run_epoch(eager=eager)
        except EarlyStop:
            pass  # On early stopping we still want to run the final traces and return results
        self._run_traces_on_end(traces=all_traces)

    def _run_epoch(self, eager: bool) -> None:
        """A method to perform an epoch of activity.

        This method requires that the current mode and epoch already be specified within the self.system object.

        Args:
            eager: Whether to run the training in eager mode. This is only related to TensorFlow training because
                PyTorch by nature is always in eager mode.
        """
        traces = get_current_items(self.traces_in_use, run_modes=self.system.mode, epoch=self.system.epoch_idx)
        trace_input_keys = set()
        for trace in traces:
            trace_input_keys.update(trace.inputs)
        network_input_keys = self.network.get_effective_input_keys(self.system.mode, self.system.epoch_idx)
        network_output_keys = self.network.get_all_output_keys(self.system.mode, self.system.epoch_idx)
        loader = self._configure_loader(
            self.pipeline.get_loader(self.system.mode,
                                     self.system.epoch_idx,
                                     output_keys=trace_input_keys - network_output_keys | network_input_keys))
        iterator = iter(loader)
        self.network.load_epoch(mode=self.system.mode,
                                epoch=self.system.epoch_idx,
                                output_keys=trace_input_keys,
                                eager=eager)
        self.system.batch_idx = None
        with Suppressor():
            batch = next(iterator)
        traces = sort_traces(traces, available_outputs=to_set(batch.keys()) | network_output_keys)
        self._run_traces_on_epoch_begin(traces=traces)
        while True:
            try:
                if self.system.mode == "train":
                    self.system.update_global_step()
                self.system.update_batch_idx()
                batch = self._configure_tensor(loader, batch)
                self._run_traces_on_batch_begin(batch, traces=traces)
                batch, prediction = self.network.run_step(batch)
                self._run_traces_on_batch_end(batch, prediction, traces=traces)
                if isinstance(loader, DataLoader) and (
                    (self.system.batch_idx == self.system.max_train_steps_per_epoch and self.system.mode == "train") or
                    (self.system.batch_idx == self.system.max_eval_steps_per_epoch and self.system.mode == "eval")):
                    raise StopIteration
                with Suppressor():
                    batch = next(iterator)
            except StopIteration:
                break
        self._run_traces_on_epoch_end(traces=traces)
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
            if isinstance(loader.dataset, OpDataset) and loader.dataset.dataset.fe_batch:
                add_batch = False
            batch = to_tensor(loader.dataset[0], target_type="tf")
            data_type = to_type(batch)
            data_shape = to_shape(batch, add_batch=add_batch, exact_shape=False)
            new_loader = tf.data.Dataset.from_generator(lambda: loader, data_type, output_shapes=data_shape)
            new_loader = new_loader.prefetch(1)
        if isinstance(new_loader, tf.data.Dataset):
            if self.system.max_train_steps_per_epoch and self.system.mode == "train":
                new_loader = new_loader.take(self.system.max_train_steps_per_epoch)
            if self.system.max_eval_steps_per_epoch and self.system.mode == "eval":
                new_loader = new_loader.take(self.system.max_eval_steps_per_epoch)
            if isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy) and isinstance(
                    self.network, TFNetwork) and not isinstance(new_loader, DistributedDataset):
                # The default autoshard policy is file, changing it to data to avoid warning
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
                new_loader = new_loader.with_options(options)
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

    def _run_traces_on_begin(self, traces: Iterable[Trace]) -> None:
        """Invoke the on_begin methods of given traces.

        Args:
            traces: List of traces.
        """
        data = Data()
        restore = None
        for trace in traces:
            # Delay RestoreWizard until the end so that it can overwrite everyone's on_begin methods
            if isinstance(trace, RestoreWizard):
                restore = trace
                continue
            # Restore does need to run before the logger though
            if isinstance(trace, Logger) and restore:
                restore.on_begin(data)
                restore = None
            trace.on_begin(data)
        if restore:
            restore.on_begin(data)
        self._check_early_exit()

    def _run_traces_on_epoch_begin(self, traces: Iterable[Trace]) -> None:
        """Invoke the on_epoch_begin methods of given traces.

        Args:
            traces: List of traces.
        """
        data = Data()
        for trace in traces:
            trace.on_epoch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_begin(self, batch: Dict[str, Any], traces: Iterable[Trace]) -> None:
        """Invoke the on_batch_begin methods of given traces.

        Args:
            batch: The batch data which was provided by the pipeline.
            traces: List of traces.
        """
        data = Data(batch)
        for trace in traces:
            trace.on_batch_begin(data)
        self._check_early_exit()

    def _run_traces_on_batch_end(self, batch: Dict[str, Any], prediction: Dict[str, Any],
                                 traces: Iterable[Trace]) -> None:
        """Invoke the on_batch_end methods of given traces.

        Args:
            batch: The batch data which was provided by the pipeline.
            prediction: The prediction data which was generated by the network.
            traces: List of traces.
        """
        data = Data(ChainMap(prediction, batch))
        for trace in traces:
            trace.on_batch_end(data)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self, traces: Iterable[Trace]) -> None:
        """Invoke the on_epoch_end methods of of given traces.

        Args:
            traces: List of traces.
        """
        data = Data()
        for trace in traces:
            trace.on_epoch_end(data)
        self._check_early_exit()

    @staticmethod
    def _run_traces_on_end(traces: Iterable[Trace]) -> None:
        """Invoke the on_end methods of given traces.

        Args:
            traces: List of traces.
        """
        data = Data()
        traceability = None
        for trace in traces:
            if isinstance(trace, Traceability):
                # Delay traceability until the end so that it can capture all data including the total training time
                traceability = trace
                continue
            trace.on_end(data)
        if traceability:
            traceability.on_end(data)

    def _check_early_exit(self) -> None:
        """Determine whether training should be prematurely aborted.

        Raises:
            EarlyStop: If the system.stop_training flag has been set to True.
        """
        if self.system.stop_training:
            raise EarlyStop


class EarlyStop(Exception):
    """An exception raised when the system.stop_training flag is flipped by a Trace in order to abort the training.

    This class is intentionally not @traceable.
    """


def enable_deterministic(seed):
    """Invoke to set random seed for deterministic training. The determinism only works for tensorflow >= 2.1 and
    pytorch >= 1.14, and some model layers don't support.

    Known failing layers:
    * tf.keras.layers.UpSampling2D

    """
    fe.fe_deterministic_seed = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
