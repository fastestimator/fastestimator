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
import functools
import gc
import multiprocessing as mp
import os
import time
from copy import deepcopy
from operator import mul
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset

from fastestimator.backend._to_tensor import to_tensor
from fastestimator.dataset.dataloader import FEDataLoader
from fastestimator.dataset.op_dataset import OpDataset, _DelayedDeepDict
from fastestimator.op.numpyop.meta.fuse import Fuse
from fastestimator.op.numpyop.meta.one_of import OneOf
from fastestimator.op.numpyop.meta.repeat import Repeat
from fastestimator.op.numpyop.meta.sometimes import Sometimes
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop, Batch
from fastestimator.schedule.schedule import Scheduler, get_current_items, EpochScheduler, \
    RepeatScheduler
from fastestimator.util.data import FilteredData
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import cpu_count, get_num_devices
from fastestimator.util.base_util import to_set, to_list

DataSource = TypeVar('DataSource', Dataset, DataLoader, tf.data.Dataset)


@traceable(blacklist=('ctx_loader', 'ctx_lock'))
class Pipeline:
    """A data pipeline class that takes care of data pre-processing.

    Args:
        train_data: The training data, or None if no training data is available.
        eval_data: The evaluation data, or None if no evaluation data is available.
        test_data: The testing data, or None if no evaluation data is available.
        batch_size: The batch size to be used by the pipeline. If the batch_size is also set by a Batch Op, that value
            will take precedence over this one (for example, if you want to set the batch_size based on mode or ds_is).
            NOTE: This argument is only applicable when using a FastEstimator Dataset.
            NOTE: This is the global batch size regardless of the number of GPUs available in the machine. If you have
                multiple (N) GPUs, each will recieve batch_size/N elements during a training step.
        ops: NumpyOps to be used for pre-processing. NOTE: This argument is only applicable when using a FastEstimator
            Dataset.
        num_process: Number of CPU threads to use for data pre-processing. NOTE: This argument is only applicable when
            using a FastEstimator Dataset. None will default to min(n_cpus, max(32, 32*n_gpus)). Multiprocessing can be
            disabled by passing 0 here, which can be useful for debugging.
    """
    ops: List[Union[NumpyOp, Scheduler[NumpyOp]]]
    data: Dict[str, Dict[Optional[str], Union[DataSource, Scheduler[DataSource]]]]  # {"mode": {"ds_id": ds}}

    def __init__(self,
                 train_data: Union[None,
                                   DataSource,
                                   Scheduler[DataSource],
                                   Dict[str, Union[DataSource, Scheduler[DataSource]]]] = None,
                 eval_data: Union[None, DataSource, Scheduler[DataSource], Dict[str, DataSource]] = None,
                 test_data: Union[None, DataSource, Scheduler[DataSource], Dict[str, DataSource]] = None,
                 batch_size: Union[None, int, Scheduler[int]] = None,
                 ops: Union[None, NumpyOp, Scheduler[NumpyOp], List[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
                 num_process: Optional[int] = None):
        data = {x: y for (x, y) in zip(["train", "eval", "test"], [train_data, eval_data, test_data]) if y}
        self.data = self._register_ds_ids(data)
        self.batch_size = batch_size
        self.ops = to_list(ops)
        if mp.get_start_method(allow_none=True) is None and os.name != 'nt':
            mp.set_start_method('fork')
        if mp.get_start_method(allow_none=True) != 'fork':
            print("FastEstimator-Warn: Pipeline multiprocessing is disabled. OS must support the 'fork' start method.")
            num_process = 0
        self.num_process = num_process if num_process is not None else min(cpu_count(), 32 * get_num_devices())
        self._verify_inputs(**{k: v for k, v in locals().items() if k != 'self'})
        # Loader Variables
        self.ctx_lock = Lock()
        self.ctx_mode = 'train'
        self.ctx_epoch = 1
        self.ctx_shuffle = True
        self.ctx_output_keys = None
        self.ctx_loader = None
        self.ctx_ds_id = None
        self.ctx_batch_size = None
        self.ctx_ops = []
        self.ctx_batch_info = Batch()
        self.ctx_batch_ops = []
        self.ctx_batch_input_keys = set()

    @staticmethod
    def _register_ds_ids(
            data: Dict[
                str, Union[DataSource, Scheduler[DataSource], Dict[str, Union[DataSource, Scheduler[DataSource]]]]]
    ) -> Dict[str, Dict[Optional[str], Union[DataSource, Scheduler[DataSource]]]]:
        """Associate dataset of each mode with a `ds_id`.

        Args:
            data: A dictionary with mode as key, dataset as value.
        """
        forbidden_ds_id_chars = {":", "!", ";", "|"}
        for mode, dataset in data.items():
            if isinstance(dataset, dict):
                for ds_name in dataset:
                    assert isinstance(ds_name, str) and len(ds_name) > 0, \
                        "dataset id must be a string, found {}".format(ds_name)
                    assert not any(char in ds_name for char in forbidden_ds_id_chars), \
                        "dataset id should not contain forbidden characters like ':', ';', '!', '|', " + \
                        "found {} in pipeline".format(ds_name)
            else:
                # Empty string is special, matches against ops which require '!ds1' but not 'ds1'
                data[mode] = {"": dataset}
        return data

    def _verify_inputs(self, **kwargs) -> None:
        """A helper method to ensure that the Pipeline inputs are valid.

        Args:
            **kwargs: A collection of variable / value pairs to validate.

        Raises:
            AssertionError: If `batch_size`, `ops`, or `num_process` were specified in the absence of a FastEstimator
                Dataset.
        """
        fe_dataset = False
        for dataset in get_current_items(set(d for ds in self.data.values() for d in ds.values())):
            fe_dataset = self._verify_dataset(dataset, **kwargs) or fe_dataset
        if not fe_dataset:
            assert kwargs['batch_size'] is None, "Pipeline only supports batch_size with built-in (FE) datasets"
            assert kwargs['ops'] is None, "Pipeline only supports ops with built-in (FE) datasets"
            assert kwargs['num_process'] is None, "Pipeline only support num_process with built-in (FE) datasets"
        # Make sure that the user provides at most 1 Batch Op for a given epoch/mode/ds_id
        batch_ops = []
        schedule_epochs = {1}
        schedule_cycles = set()
        for op in self.ops:
            if isinstance(op, Batch):
                batch_ops.append(op)
            if isinstance(op, Scheduler):
                # Only keep the scheduler if it contains at least one Batch op
                vals = op.get_all_values()
                for val in vals:
                    if isinstance(val, Batch):
                        batch_ops.append(op)
                        if isinstance(op, EpochScheduler):
                            schedule_epochs |= op.epoch_dict.keys()
                        elif isinstance(op, RepeatScheduler):
                            schedule_cycles.add(op.cycle_length)
                        else:
                            # Some unknown scheduler, no known shortcuts so just try first 100 epochs to be safe
                            schedule_epochs |= {*range(1, 100)}
                        break
        # After m*n steps all possible m and n combinations will be visited
        schedule_cycles = functools.reduce(mul, schedule_cycles, 1)
        # Consider x + m*n epochs for each epoch scheduler x value
        schedule_epochs = sorted({epoch for base_epoch in schedule_epochs for epoch in
                                  list(range(base_epoch, base_epoch + schedule_cycles))})
        for mode, id_ds in self.data.items():
            for ds_id in id_ds.keys():
                for epoch in schedule_epochs:
                    ops = get_current_items(batch_ops, run_modes=mode, epoch=epoch, ds_id=ds_id)
                    # We have to do an instance check again since the user could technically use a scheduler that has a
                    # Batch Op at one point, but some other Op (or None) at a different point
                    ops = [op for op in ops if isinstance(op, Batch)]
                    assert len(ops) < 2, "You may provide at most 1 batch op for a given epoch/mode/ds_id combination"

    def _verify_dataset(self, dataset: DataSource, **kwargs) -> bool:
        """A helper function to ensure that all of a dataset's arguments are correct.

        Args:
            dataset: The dataset to validate against.
            **kwargs: A selection of variables and their values which must be validated.

        Returns:
            True iff the `dataset` is a PyTorch Dataset (as opposed to a DataLoader or tf.data.Dataset).

        Raises:
            AssertionError: If the `kwargs` are found to be invalid based on the given `dataset`.
            ValueError: If the `dataset` is of an unknown type.
        """
        if isinstance(dataset, Dataset):
            # batch_size check
            for batch_size in get_current_items(to_list(self.batch_size)):
                assert isinstance(batch_size, int), "unsupported batch_size format: {}".format(type(batch_size))
            # ops check
            for op in get_current_items(self.ops):
                assert isinstance(op, NumpyOp), "unsupported op format, must provide NumpyOp in Pipeline"
            # num_process check
            assert isinstance(self.num_process, int), "number of processes must be an integer"
            return True
        elif isinstance(dataset, (DataLoader, tf.data.Dataset)):
            if kwargs['batch_size'] is not None:
                print("FastEstimator-Warn: batch_size will only be used for built-in dataset")
            if kwargs['ops'] is not None:
                print("FastEstimator-Warn: ops will only be used for built-in dataset")
            if kwargs['num_process'] is not None:
                print("FastEstimator-Warn: num_process will only be used for built-in dataset")
            return False
        else:
            raise ValueError("Unsupported dataset type: {}".format(type(dataset)))

    def _get_op_split(self, mode: str, epoch: int, ds_id: str) -> Tuple[List[NumpyOp], Batch, List[NumpyOp]]:
        """Figure out which ops are pre-batch vs post-batch.

        Args:
            mode: The current mode.
            epoch: The current epoch.
            ds_id: The current dataset.

        Returns:
            (instance ops, batch info, batch ops).
        """
        batch_info = Batch()
        instance_ops = []
        batch_ops = []
        ops = get_current_items(self.ops, run_modes=mode, epoch=epoch, ds_id=ds_id)
        target = instance_ops
        for op in ops:
            if isinstance(op, Batch):
                batch_info = op
                target = batch_ops
                continue
            target.append(op)
        return instance_ops, batch_info, batch_ops

    def get_modes(self, epoch: Optional[int] = None) -> Set[str]:
        """Get the modes for which the Pipeline has data.

        Args:
            epoch: The current epoch index

        Returns:
            The modes for which the Pipeline has data.
        """
        if epoch is None:
            all_modes = set(self.data.keys())
        else:
            all_modes = []
            for mode, datasets in self.data.items():
                for dataset in datasets.values():
                    if isinstance(dataset, Scheduler):
                        dataset = dataset.get_current_value(epoch)
                    if dataset:
                        all_modes.append(mode)
        return to_set(all_modes)

    def get_ds_ids(self, epoch: int, mode: str) -> List[Union[str, None]]:
        """Get the ds_ids for a given epoch and mode.

        Args:
            epoch: The current epoch index.
            mode: The current execution mode.

        Returns:
            The ds_ids of the current epoch and mode.
        """
        ds_ids = []
        if mode in self.data:
            datasets = self.data[mode]
            for ds_id, dataset in datasets.items():
                if isinstance(dataset, Scheduler):
                    dataset = dataset.get_current_value(epoch)
                if dataset:
                    ds_ids.append(ds_id)
        return ds_ids

    def benchmark(self,
                  mode: str = "train",
                  epoch: int = 1,
                  ds_id: Optional[str] = None,
                  num_steps: int = 1000,
                  log_interval: int = 100,
                  detailed: bool = True) -> None:
        """Benchmark the pipeline processing speed.

        Args:
            mode: The execution mode to benchmark. This can be 'train', 'eval' or 'test'.
            epoch: The epoch index to benchmark. Note that epoch indices are 1-indexed.
            ds_id: The ds_id to benchmark. If None, all ds_ids will be benchmarked.
            num_steps: The number of steps over which to perform the benchmark.
            log_interval: The logging interval.
            detailed: Whether to display the detailed time used by each operator.
        """
        if ds_id is None:
            ds_ids = self.get_ds_ids(epoch=epoch, mode=mode)
        else:
            ds_ids = [ds_id]

        for ds_id in ds_ids:
            with self(mode=mode, epoch=epoch, ds_id=ds_id, steps_per_epoch=num_steps) as loader:
                if isinstance(loader, tf.data.Dataset):
                    loader = loader.take(num_steps)
                start = time.perf_counter()
                for idx, _ in enumerate(loader, start=1):
                    if idx % log_interval == 0:
                        duration = time.perf_counter() - start
                        iters_per_sec = log_interval / duration
                        ds_str = f"Dataset: {ds_id}, " if ds_id else ""
                        print("FastEstimator-Benchmark ({}): {}Step: {}, Epoch: {}, Steps/sec: {}".format(
                            mode.capitalize(), ds_str, idx, epoch, iters_per_sec))
                        start = time.perf_counter()
                # Pipeline Operations Benchmarking when using FEDataset
                if isinstance(loader, FEDataLoader) and isinstance(loader.dataset, OpDataset) and detailed:
                    # (n_visited, duration)
                    duration_list = np.zeros(shape=(len(self.ctx_ops) + 1 + len(self.ctx_batch_ops), 2))
                    data_len = len(loader.dataset)
                    ds_str = f", Dataset: {ds_id}" if ds_id else ""
                    print("\nBreakdown of time taken by Pipeline Operations (Mode: {}, Epoch: {}{})\n".format(
                        mode.capitalize(), epoch, ds_str))
                    extra_memory_management_time = 0
                    for _ in range(log_interval):
                        filtered = False
                        batch = []
                        index = np.random.randint(data_len)
                        items = deepcopy(loader.dataset.dataset[index])
                        if isinstance(items, list):
                            while not batch:
                                filtered = False
                                # BatchDataset may randomly sample the same elements multiple times, avoid reprocessing
                                unique_samples = set()
                                for item in items:
                                    if id(item) not in unique_samples:
                                        for i, op in enumerate(self.ctx_ops):
                                            start = time.perf_counter()
                                            op_data = forward_numpyop([op], item, {'mode': loader.dataset.mode})
                                            duration = time.perf_counter() - start
                                            duration_list[i][0] += 1
                                            duration_list[i][1] += duration
                                            if isinstance(op_data, FilteredData):
                                                filtered = True
                                                break
                                        unique_samples.add(id(item))
                                if not filtered:
                                    batch = items
                        else:
                            while len(batch) < (self.ctx_batch_size or 1):
                                filtered = False
                                for i, op in enumerate(self.ctx_ops):
                                    start = time.perf_counter()
                                    op_data = forward_numpyop([op], items, {'mode': mode})
                                    duration = time.perf_counter() - start
                                    duration_list[i][0] += 1
                                    duration_list[i][1] += duration
                                    if isinstance(op_data, FilteredData):
                                        filtered = True
                                        break
                                if not filtered:
                                    batch.append(items)
                                index = np.random.randint(data_len)
                                items = deepcopy(loader.dataset.dataset[index])
                        if not filtered:
                            # Perform the batching
                            start = time.perf_counter()
                            batch = self.ctx_batch_info.collate_fn(batch)
                            duration = time.perf_counter() - start
                            duration_list[len(self.ctx_ops)][0] += 1
                            duration_list[len(self.ctx_ops)][1] += duration
                            # Perform batch ops
                            start = time.perf_counter()
                            # Transform to numpy to not bias against the first op in the batch_op chain
                            batch = to_tensor(batch, target_type='np')
                            extra_memory_management_time += time.perf_counter() - start

                            for i, op in enumerate(self.ctx_batch_ops, start=len(self.ctx_ops) + 1):
                                start = time.perf_counter()
                                op_data = forward_numpyop([op], data=batch, state={'mode': mode}, batched='np')
                                duration = time.perf_counter() - start
                                duration_list[i][0] += 1
                                duration_list[i][1] += duration
                                if isinstance(op_data, FilteredData):
                                    break
                            # Count extra time needed to cast data back to torch
                            start = time.perf_counter()
                            to_tensor(batch, target_type='torch', shared_memory=True)
                            extra_memory_management_time += time.perf_counter() - start

                    if self.ctx_batch_ops:
                        # Extra memory management penalty is only incurred when using batch ops
                        duration_list[len(self.ctx_ops)][1] += extra_memory_management_time

                    total_time = np.sum(duration_list[:, 1])
                    normalized_times_ms = 1000 * duration_list[:, 1] / np.maximum(duration_list[:, 0], 1)
                    op_names = ["Op"]

                    for op in self.ctx_ops + [self.ctx_batch_info] + self.ctx_batch_ops:
                        if isinstance(op, Sometimes) and op.op:
                            op_names.append(op.__class__.__name__ + " (" + op.op.__class__.__name__ + ")")
                        elif isinstance(op, Repeat) and op.op:
                            op_names.append(op.__class__.__name__ + " (" + op.op.__class__.__name__ + ")")
                        elif isinstance(op, OneOf) and op.ops:
                            op_names.append(op.__class__.__name__ + " (" +
                                            ", ".join([sub_op.__class__.__name__ for sub_op in op.ops]) + ")")
                        elif isinstance(op, Fuse) and op.ops:
                            op_names.append(op.__class__.__name__ + " (" +
                                            ", ".join([sub_op.__class__.__name__ for sub_op in op.ops]) + ")")
                        elif isinstance(op, Batch):
                            op_names.append("<Collating Batch>")
                        else:
                            op_names.append(op.__class__.__name__)

                    max_op_len = max(len(op_name) for op_name in op_names)
                    max_in_len = max([len(", ".join(op.inputs)) for op in
                                      self.ctx_ops + [self.ctx_batch_info] + self.ctx_batch_ops] + [len("Inputs")])
                    max_out_len = max([len(", ".join(op.outputs)) for op in
                                       self.ctx_ops + [self.ctx_batch_info] + self.ctx_batch_ops] + [len("Outputs")])
                    ms_visit_len = max(len("{:.3f}".format(max(normalized_times_ms))), len("ms / Visit"))
                    visit_len = max(len(f"{int(np.max(duration_list[:, 0]))}"), len("Visits"))

                    print("{}: {}: {}: {}: {}: {}".format("Op".ljust(max_op_len + 1),
                                                          "Inputs".ljust(max_in_len + 1),
                                                          "Outputs".ljust(max_out_len + 1),
                                                          "ms / Visit".ljust(ms_visit_len + 1),
                                                          "Visits".ljust(visit_len + 1),
                                                          "Time (Total)".rjust(12)))
                    print("-" * (max_op_len + max_in_len + max_out_len + visit_len + 37))
                    for i, op in enumerate(self.ctx_ops + [self.ctx_batch_info] + self.ctx_batch_ops):
                        print("{}: {}: {}: {}: {}: {:11.2f}%".format(
                            op_names[i + 1].ljust(max_op_len + 1),
                            ", ".join(op.inputs).ljust(max_in_len + 1),
                            ", ".join(op.outputs).ljust(max_out_len + 1),
                            "{:.3f}".format(normalized_times_ms[i]).ljust(ms_visit_len + 1),
                            str(int(duration_list[i][0])).ljust(visit_len + 1),
                            100 * duration_list[i][1] / total_time))
                    if self.ctx_batch_ops:
                        penalty = round(100*(duration_list[len(self.ctx_ops)][1] - extra_memory_management_time) /
                                        duration_list[len(self.ctx_ops)][1], 1)
                        print(f"\nNote that collation time would be cut by ~{penalty}% if there were no batched ops.")
                print("\n")  # to make printing more obvious

    def get_scheduled_items(self, mode: str) -> List[Any]:
        """Get a list of items considered for scheduling.

        Args:
            mode: Current execution mode.

        Returns:
            List of schedulable items in Pipeline.
        """
        all_items = self.ops + [self.batch_size] + list(self.data[mode].values())
        return all_items

    def get_epochs_with_data(self, total_epochs: int, mode: str) -> Set[int]:
        """Get a set of epoch indices that contains data given mode.

        Args:
            total_epochs: Total number of epochs.
            mode: Current execution mode.

        Returns:
            Set of epoch indices.
        """
        epochs_with_data = set()
        datasets = self.data[mode]
        for dataset in datasets.values():
            if isinstance(dataset, Scheduler):
                epochs_with_data_ds = set(epoch for epoch in range(1, total_epochs + 1)
                                          if dataset.get_current_value(epoch))
                epochs_with_data = epochs_with_data | epochs_with_data_ds
            elif dataset:
                epochs_with_data_ds = set(range(1, total_epochs + 1))
                epochs_with_data = epochs_with_data | epochs_with_data_ds
                break
        return epochs_with_data

    def transform(self,
                  data: Dict[str, Any],
                  mode: str,
                  epoch: int = 1,
                  ds_id: str = '',
                  target_type: str = 'np') -> Union[Dict[str, Any], FilteredData]:
        """Apply all pipeline operations on a given data instance for the specified `mode` and `epoch`.

        Args:
            data: Input data in dictionary format.
            mode: The execution mode in which to run. This can be "train", "eval", "test" or "infer".
            epoch: The epoch index to run. Note that epoch indices are 1-indexed.
            ds_id: The current dataset id.
            target_type: What kind of tensor(s) to create. One of "tf", "torch", or "np".

        Returns:
            The transformed data.
        """
        data = deepcopy(data)
        instance_ops, batch_spec, batch_ops = self._get_op_split(mode=mode, epoch=epoch, ds_id=ds_id)
        state = {'mode': mode}
        op_data = forward_numpyop(instance_ops, data, state)
        if isinstance(op_data, FilteredData):
            return op_data
        data = batch_spec.collate_fn([data])
        op_data = forward_numpyop(batch_ops, data, state, batched='torch')
        if isinstance(op_data, FilteredData):
            return op_data
        return to_tensor(data, target_type=target_type)

    def get_results(self,
                    mode: str = "train",
                    epoch: int = 1,
                    ds_id: str = '',
                    num_steps: int = 1,
                    shuffle: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Get sample Pipeline outputs.

        Args:
            mode: The execution mode in which to run. This can be "train", "eval", or "test".
            epoch: The epoch index to run. Note that epoch indices are 1-indexed.
            num_steps: Number of steps (batches) to get.
            shuffle: Whether to use shuffling.
            ds_id: The current dataset id.

        Returns:
            A list of batches of Pipeline outputs.
        """
        results = []
        with self(mode=mode, epoch=epoch, ds_id=ds_id, shuffle=shuffle) as loader:
            if isinstance(loader, tf.data.Dataset):
                loader = loader.take(num_steps)
            if loader:
                for idx, batch in enumerate(loader, start=1):
                    results.append(batch)
                    if idx == num_steps:
                        break
                if len(results) == 1:
                    results = results[0]
            return results

    def __call__(self,
                 mode: str,
                 epoch: int = 1,
                 ds_id: str = '',
                 shuffle: Optional[bool] = None,
                 steps_per_epoch: Optional[int] = None,
                 output_keys: Optional[Set[str]] = None) -> 'Pipeline':
        """Prepare this Pipeline for a given `mode` and `epoch`.

        A given pipeline can only provide one loader at a time. This helps to prevent issues with multi-threading.

        ```python
        pipe = Pipeline(...)
        with pipe(mode='eval', epoch=2) as loader:
            for batch in loader:
                print(batch)
        ```

        Args:
            mode: The execution mode for the loader. This can be 'train', 'eval' or 'test'.
            epoch: The epoch index for the loader. Note that epoch indices are 1-indexed.
            ds_id: The dataset id to consider for the loader.
            shuffle: Whether to shuffle the data. If None, the value for shuffle is based on mode. NOTE: This argument
                is only used with FastEstimator Datasets.
            steps_per_epoch: Training or Evaluation will be cut short or extended to complete N steps even if loader is
                not yet exhausted. If None, all data will be used.
            output_keys: What keys can be produced from pipeline. If None or empty, all keys will be considered.

        Returns:
            The pipeline, but with `mode` and `epoch` set for use in a loader.

        Raises:
            ValueError: If called while the pipeline already has an active loader.
        """
        # Make sure that a loader isn't currently instantiated with other settings
        acquired = self.ctx_lock.acquire(blocking=False)
        if not acquired:
            raise ValueError("You cannot invoke a Pipeline's __call__ method while it already has an active loader.")
        self.ctx_mode = mode
        self.ctx_epoch = epoch
        self.ctx_ds_id = ds_id
        self.ctx_shuffle = mode == 'train' if shuffle is None else shuffle
        self.ctx_steps_per_epoch = steps_per_epoch
        self.ctx_output_keys = output_keys or set()
        self.ctx_ops, self.ctx_batch_info, self.ctx_batch_ops = self._get_op_split(mode=mode, epoch=epoch, ds_id=ds_id)
        # Figure out which input keys are required by the batch ops (so they don't get pruned too early)
        self.ctx_batch_input_keys = set()
        batch_produced_keys = set()
        for op in get_current_items(self.ctx_batch_ops, mode, epoch, ds_id=ds_id):
            self.ctx_batch_input_keys.update(set(key for key in op.inputs if key not in batch_produced_keys))
            batch_produced_keys.update(op.outputs)
        # Decide on the batch size (this might still be ignored later if the user is using a BatchDataset)
        self.ctx_batch_size = self.ctx_batch_info.batch_size
        if self.ctx_batch_size is None:
            # batch size
            batch_size = self.batch_size
            if isinstance(batch_size, Scheduler):
                batch_size = batch_size.get_current_value(self.ctx_epoch)
            self.ctx_batch_size = batch_size
        self.ctx_lock.release()
        return self

    def __enter__(self) -> Union[DataLoader, tf.data.Dataset]:
        """Get a data loader from the Pipeline for the current epoch and mode.

        A given pipeline can only provide one loader at a time. This helps to prevent issues with multi-threading.

        ```python
        pipe = Pipeline(...)
        with pipe(mode='eval', epoch=2) as loader:
            for batch in loader:
                print(batch)
        ```

        Returns:
            A data loader for the current `mode` and `epoch`.

        Raises:
            ValueError: If called while the pipeline already has an active loader.
        """
        acquired = self.ctx_lock.acquire(blocking=False)
        if not acquired:
            raise ValueError("You cannot generate a new loader from this Pipeline before closing its other loader.")
        # Release the lock if arguments are invalid so that people in Jupyter / debug consoles don't get stuck
        if self.ctx_mode not in self.data:
            self.ctx_lock.release()
            raise KeyError(f"Pipeline has no data for mode '{self.ctx_mode}'")
        if self.ctx_ds_id not in self.data[self.ctx_mode]:
            self.ctx_lock.release()
            raise KeyError(f"The dataset id '{self.ctx_ds_id}' is not present in {self.ctx_mode} mode")
        data = self.data[self.ctx_mode][self.ctx_ds_id]
        if isinstance(data, Scheduler):
            data = data.get_current_value(self.ctx_epoch)
        if isinstance(data, Dataset):
            # Results will be immediately converted to tensors, so don't need deep_remainder
            op_dataset = OpDataset(data,
                                   self.ctx_ops,
                                   self.ctx_mode,
                                   self.ctx_output_keys | self.ctx_batch_input_keys if self.ctx_output_keys else None,
                                   deep_remainder=False)
            # check whether to batch the data
            batch_size = None if op_dataset.fe_batch else self.ctx_batch_size
            # Figure out whether a postprocessing function is needed (for batched ops)
            postprocess_fn = None
            if self.ctx_batch_ops:
                postprocess_fn = functools.partial(_batch_postprocess,
                                                   ops=self.ctx_batch_ops,
                                                   output_keys=self.ctx_output_keys,
                                                   mode=self.ctx_mode)
            try:
                data = FEDataLoader(op_dataset,
                                    postprocess_fn=postprocess_fn,
                                    batch_size=batch_size,
                                    shuffle=self.ctx_shuffle,
                                    steps_per_epoch=self.ctx_steps_per_epoch,
                                    num_workers=self.num_process,
                                    drop_last=self.ctx_batch_info.drop_last,
                                    collate_fn=self.ctx_batch_info.collate_fn)
            except ValueError as err:
                self.ctx_lock.release()
                raise err
            self.ctx_loader = data
        return data

    def __exit__(self, *exc: Tuple[Optional[Type], Optional[Exception], Optional[Any]]) -> None:
        if self.ctx_loader is not None:
            self.ctx_loader.shutdown()
            self.ctx_loader = None
        # Manually triggering gc here seems to be necessary in order to avoid problems with repeated invocations of FE
        # killing one another through multi-processing.
        gc.collect()
        self.ctx_lock.release()


def _batch_postprocess(data: Dict[str, Any], ops: List[NumpyOp], output_keys: Set[str], mode: str) -> \
        Union[Dict[str, Any], FilteredData]:
    op_data = forward_numpyop(ops=ops, data=data, state={'mode': mode}, batched='torch')
    if isinstance(op_data, FilteredData):
        return op_data
    if output_keys:
        for key in data.keys() - output_keys:
            if key not in _DelayedDeepDict.warned:
                _DelayedDeepDict.warned.add(key)
                print("FastEstimator-Warn: the key '{}' is being pruned since it is unused outside of the Pipeline."
                      " To prevent this, you can declare the key as an input of a Trace or TensorOp.".format(key))
            data.pop(key)
    return data
