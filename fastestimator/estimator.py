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
import inspect
import os
import random
from collections import ChainMap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.distribute.input_lib import DistributedDataset
from torch.utils.data import DataLoader

import fastestimator as fe
from fastestimator.backend._to_shape import to_shape
from fastestimator.backend._to_tensor import to_tensor
from fastestimator.backend._to_type import to_type
from fastestimator.network import BaseNetwork, TFNetwork, TorchNetwork
from fastestimator.pipeline import Pipeline
from fastestimator.schedule.schedule import Scheduler, get_current_items, get_signature_epochs
from fastestimator.summary.history import HistoryRecorder
from fastestimator.summary.system import Summary, System
from fastestimator.trace.io.best_model_saver import BestModelSaver
from fastestimator.trace.io.model_saver import ModelSaver
from fastestimator.trace.io.restore_wizard import RestoreWizard
from fastestimator.trace.io.traceability import Traceability
from fastestimator.trace.trace import EvalEssential, Logger, PerDSTrace, TestEssential, Trace, TrainEssential, \
    sort_traces
from fastestimator.util.base_util import NonContext, Suppressor, to_list, to_set
from fastestimator.util.data import Data, FilteredData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import draw


def _verify_dependency_versions() -> None:
    """Print warning messages if the user is using unexpected versions of TF or torch.
    """
    if tf.__version__ != '2.9.1':
        print("\033[93m{}\033[00m".format("FastEstimator-Warn: Expected TensorFlow version 2.9.1 but found "
                                          f"{tf.__version__}. The framework may not work as expected."))
    if torch.__version__ not in ('1.10.2', '1.10.2+cpu', '1.10.2+cu113'):
        print("\033[93m{}\033[00m".format("FastEstimator-Warn: Expected PyTorch version 1.10.2 but found "
                                          f"{torch.__version__}. The framework may not work as expected."))


@traceable()
class Estimator:
    """One class to rule them all.

    Estimator is the highest level class within FastEstimator. It is the class which is invoked to actually train
    (estimator.fit) or test (estimator.test) models. It wraps `Pipeline`, `Network`, `Trace` objects together and
    defines the whole optimization process.

    If the data fed into pipeline is a TensorFlow Dataset, then the parameters `train_steps_per_epoch` and
    `eval_steps_per_epoch` can only reduce the number of steps per epoch. If these parameters are higher than the
    dimension of the stated Dataset then the whole Dataset will be used.


    Args:
        pipeline: An fe.Pipeline object that defines the data processing workflow.
        network: An fe.Network object that contains models and other training graph definitions.
        epochs: The number of epochs to run.
        train_steps_per_epoch: Training will be cut short or extended to complete N steps even if loader is not yet
            exhausted. If None, all data will be used.
        eval_steps_per_epoch: Evaluation will be cut short or extended to complete N steps even if loader is not yet
            exhausted. If None, all data will be used.
        traces: What Traces to run during training. If None, only the system's default Traces will be included.
        log_steps: Frequency (in steps) for printing log messages. 0 to disable all step-based printing (though epoch
            information will still print). None to completely disable printing.
        eval_log_steps: The list of steps on which evaluation progress logs need to be printed.
        monitor_names: Additional keys from the data dictionary to be written into the logs.
    """
    monitor_names: Set[str]
    traces_in_use: List[Union[Trace, Scheduler[Trace]]]
    system: System
    filepath: str

    def __init__(self,
                 pipeline: Pipeline,
                 network: BaseNetwork,
                 epochs: int,
                 train_steps_per_epoch: Optional[int] = None,
                 eval_steps_per_epoch: Optional[int] = None,
                 traces: Union[None, Trace, Scheduler[Trace], Iterable[Union[Trace, Scheduler[Trace]]]] = None,
                 log_steps: Optional[int] = 100,
                 eval_log_steps: Sequence[int] = (),
                 monitor_names: Union[None, str, Iterable[str]] = None):
        self.traces_in_use = []
        self.filepath = os.path.realpath(inspect.stack()[2].filename)  # Record this for history tracking
        assert log_steps is None or log_steps >= 0, \
            "log_steps must be None or positive (or 0 to disable only train logging)"
        self.monitor_names = to_set(monitor_names) | network.get_loss_keys()
        self.system = System(network=network,
                             pipeline=pipeline,
                             traces=to_list(traces),
                             log_steps=log_steps,
                             total_epochs=epochs,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch,
                             eval_log_steps=eval_log_steps,
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
        _verify_dependency_versions()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Prevent tf from constantly printing useless information
        draw()
        self.system.reset(summary, self.fe_summary())
        self._prepare_traces(run_modes={"train", "eval"})
        if warmup:
            self._warmup(eager=eager)
        self._start(run_modes={"train", "eval"}, eager=eager)
        return self.system.summary or None

    def _prepare_traces(self, run_modes: Set[str]) -> None:
        """Prepare information about the traces for execution.

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
        for trace in sort_traces(get_current_items(self.traces_in_use, run_modes=run_modes), ds_ids=[]):
            trace_outputs.update(trace.get_outputs(ds_ids=[]))
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
        _verify_dependency_versions()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Prevent tf from constantly printing useless information
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
        sort_traces(all_traces, ds_ids=[])  # This ensures that the traces can sort properly for on_begin and on_end
        monitor_names = self.monitor_names
        for mode in self.pipeline.get_modes() - {"test"}:
            scheduled_items = self.pipeline.get_scheduled_items(mode) + self.network.get_scheduled_items(
                mode) + self.get_scheduled_items(mode)
            signature_epochs = get_signature_epochs(scheduled_items, self.system.total_epochs, mode=mode)
            epochs_with_data = self.pipeline.get_epochs_with_data(total_epochs=self.system.total_epochs, mode=mode)
            for epoch in signature_epochs:
                if epoch not in epochs_with_data:
                    continue
                ds_ids = self.pipeline.get_ds_ids(epoch, mode)
                for ds_id in ds_ids:
                    network_output_keys = self.network.get_all_output_keys(mode, epoch, ds_id=ds_id)
                    network_input_keys = self.network.get_effective_input_keys(mode, epoch, ds_id=ds_id)
                    trace_input_keys = set()
                    trace_output_keys = {"*"}
                    traces = get_current_items(self.traces_in_use, run_modes=mode, epoch=epoch, ds_id=ds_id)
                    for idx, trace in enumerate(traces):
                        if idx > 0:  # ignore TrainEssential and EvalEssential's inputs for unmet requirement checking
                            trace_input_keys.update(trace.inputs)
                        trace_output_keys.update(trace.get_outputs(ds_ids=ds_ids))
                    # key checking
                    with self.pipeline(mode=mode,
                                       epoch=epoch,
                                       ds_id=ds_id,
                                       steps_per_epoch=None,
                                       output_keys=trace_input_keys - network_output_keys
                                       | network_input_keys) as loader:
                        loader = self._configure_loader(loader)
                        with Suppressor():
                            if isinstance(loader, tf.data.Dataset):
                                batch = list(loader.take(1))[0]
                            else:
                                batch = next(iter(loader))
                        batch = self._configure_tensor(loader, batch)
                    assert isinstance(batch, dict), "please make sure data output format is dictionary"
                    pipeline_output_keys = to_set(batch.keys())

                    monitor_names = monitor_names - (pipeline_output_keys | network_output_keys)
                    unmet_requirements = trace_input_keys - (pipeline_output_keys | network_output_keys
                                                             | trace_output_keys)
                    assert not unmet_requirements, \
                        "found missing key(s) during epoch {} mode {} ds_id {}: {}".format(epoch, mode, ds_id,
                                                                                           unmet_requirements)
                    sort_traces(traces, ds_ids=ds_ids, available_outputs=pipeline_output_keys | network_output_keys)
                    trace_input_keys.update(traces[0].inputs)
                    self.network.load_epoch(mode, epoch, ds_id, output_keys=trace_input_keys, warmup=True, eager=eager)
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
        all_traces = sort_traces(get_current_items(self.traces_in_use, run_modes=run_modes), ds_ids=[])
        with NonContext() if fe.fe_history_path is False else HistoryRecorder(
                self.system, self.filepath, db_path=fe.fe_history_path):
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
        ds_ids = self.pipeline.get_ds_ids(self.system.epoch_idx, self.system.mode)
        epoch_traces = sort_traces(
            get_current_items(self.traces_in_use, run_modes=self.system.mode, epoch=self.system.epoch_idx),
            ds_ids=ds_ids)
        self._run_traces_on_epoch_begin(traces=epoch_traces)
        self.system.batch_idx = None
        end_epoch_data = Data()  # We will aggregate data over on_ds_end and put it into on_epoch_end for printing
        # run for each dataset
        for self.system.ds_id in ds_ids:
            ds_traces = get_current_items(self.traces_in_use,
                                          run_modes=self.system.mode,
                                          epoch=self.system.epoch_idx,
                                          ds_id=self.system.ds_id)
            trace_input_keys = set()
            for ds_trace in ds_traces:
                trace_input_keys.update(ds_trace.inputs)
            network_input_keys = self.network.get_effective_input_keys(mode=self.system.mode,
                                                                       epoch=self.system.epoch_idx,
                                                                       ds_id=self.system.ds_id)
            network_output_keys = self.network.get_all_output_keys(mode=self.system.mode,
                                                                   epoch=self.system.epoch_idx,
                                                                   ds_id=self.system.ds_id)
            self.network.load_epoch(mode=self.system.mode,
                                    epoch=self.system.epoch_idx,
                                    ds_id=self.system.ds_id,
                                    output_keys=trace_input_keys,
                                    eager=eager)

            with self.pipeline(mode=self.system.mode,
                               epoch=self.system.epoch_idx,
                               ds_id=self.system.ds_id,
                               steps_per_epoch=self.system.steps_per_epoch,
                               output_keys=trace_input_keys - network_output_keys | network_input_keys) as loader:

                if self.system.mode == 'eval':
                    log_steps_per_epoch = len(loader) // loader.get_batch_size(
                    ) if not self.system.steps_per_epoch else self.system.steps_per_epoch
                    self.system.eval_log_steps = ([
                        1, log_steps_per_epoch // 3, (2 * log_steps_per_epoch) // 3, log_steps_per_epoch
                    ], log_steps_per_epoch) if not self.system.eval_log_steps_request else \
                        (self.system.eval_log_steps_request, log_steps_per_epoch)

                loader = self._configure_loader(loader)
                iterator = iter(loader)
                with Suppressor():
                    batch = next(iterator)
                ds_traces = sort_traces(ds_traces,
                                        available_outputs=to_set(batch.keys()) | network_output_keys,
                                        ds_ids=ds_ids)
                per_ds_traces = [trace for trace in ds_traces if isinstance(trace, PerDSTrace)]
                self._run_traces_on_ds_begin(traces=per_ds_traces)
                while True:
                    try:
                        if self.system.mode == "train":
                            self.system.update_global_step()
                        self.system.update_batch_idx()
                        batch = self._configure_tensor(loader, batch)
                        self._run_traces_on_batch_begin(batch, traces=ds_traces)

                        batch, prediction = self.network.run_step(batch)
                        self._run_traces_on_batch_end(batch, prediction, traces=ds_traces)
                        if isinstance(loader, DataLoader) and (
                            (self.system.batch_idx == self.system.train_steps_per_epoch and self.system.mode == "train")
                                or
                            (self.system.batch_idx == self.system.eval_steps_per_epoch and self.system.mode == "eval")):
                            raise StopIteration
                        with Suppressor():
                            batch = next(iterator)
                    except StopIteration:
                        break
                self._run_traces_on_ds_end(traces=per_ds_traces, data=end_epoch_data)
            self.network.unload_epoch()
        self._run_traces_on_epoch_end(traces=epoch_traces, data=end_epoch_data)

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
            add_batch = bool(new_loader.batch_size)
            if hasattr(loader, 'fe_postprocess_fn') and loader.fe_postprocess_fn is not None:
                # The user is manually batching data and running ops on data batches. No reliable way to shortcut this
                # since ops might require specific batch composition.
                data_instance = next(iter(loader))
                add_batch = False
            else:
                # No batch-based ops so we can try and just use the OpDataset to more quickly get our data summary
                data_instance = loader.dataset[0]
                if isinstance(data_instance, list):
                    # This is a batched dataset
                    data_instance = data_instance[0]
                    add_batch = True
                if isinstance(data_instance, FilteredData):
                    # We got unlucky and drew filtered data as the zeroth element. Fall back to a slower but more robust
                    # analysis of the batch
                    data_instance = next(iter(loader))
                    add_batch = False
            data_instance = to_tensor(data_instance, target_type="tf")
            data_type = to_type(data_instance)
            data_shape = to_shape(data_instance, add_batch=add_batch, exact_shape=False)
            new_loader = tf.data.Dataset.from_generator(lambda: loader, data_type, output_shapes=data_shape)
            new_loader = new_loader.prefetch(1)
        if isinstance(new_loader, tf.data.Dataset):
            if self.system.train_steps_per_epoch and self.system.mode == "train":
                new_loader = new_loader.take(self.system.train_steps_per_epoch)
            if self.system.eval_steps_per_epoch and self.system.mode == "eval":
                new_loader = new_loader.take(self.system.eval_steps_per_epoch)
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
        # TODO - if user has torch loader but custom collate that doesn't return torch tensor, need to cast here
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

    def _run_traces_on_ds_begin(self, traces: Iterable[PerDSTrace]) -> None:
        """Invoke the on_ds_begin methods of given traces.

        Args:
            traces: List of traces.
        """
        data = Data()
        for trace in traces:
            trace.on_ds_begin(data)
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

    def _run_traces_on_ds_end(self, traces: Iterable[PerDSTrace], data: Data) -> None:
        """Invoke the on_ds_begin methods of given traces.

        Args:
            traces: List of traces.
            data: Data into which to record results.
        """
        for trace in traces:
            trace.on_ds_end(data)
        self._check_early_exit()

    def _run_traces_on_epoch_end(self, traces: Iterable[Trace], data: Data) -> None:
        """Invoke the on_epoch_end methods of of given traces.

        Args:
            traces: List of traces.
            data: Data into which to record results.
        """
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


def enable_deterministic(seed: int) -> None:
    """Invoke to set random seed for deterministic training.

    The determinism only works for tensorflow >= 2.1 and pytorch >= 1.14, and some model layers don't support.

    Known failing layers:
    * tf.keras.layers.UpSampling2D

    Args:
        seed: The random seed to use for training.
    """
    fe.fe_deterministic_seed = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def record_history(path: Union[bool, str]) -> None:
    """Change the default location for history tracking.

    Args:
        path: The path to save experiment histories. Pass True to use the default location of
            ~/fastestimator_data/history.db. Pass False to disable history tracking.
    """
    if path in (None, True):
        fe.fe_history_path = None
    else:
        fe.fe_history_path = path
