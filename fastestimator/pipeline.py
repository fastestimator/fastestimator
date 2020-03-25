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
import time
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset, RandomSampler

from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.op.op import get_current_ops
from fastestimator.schedule.schedule import EpochScheduler, RepeatScheduler, Scheduler
from fastestimator.util.util import lcms, to_list

DataSource = TypeVar('DataSource', Dataset, DataLoader, tf.data.Dataset)


class Pipeline:
    ops: List[Union[NumpyOp, Scheduler[NumpyOp]]]

    def __init__(self,
                 train_data: Union[None, DataSource, Scheduler[DataSource]] = None,
                 eval_data: Union[None, DataSource, Scheduler[DataSource]] = None,
                 test_data: Union[None, DataSource, Scheduler[DataSource]] = None,
                 batch_size: Union[None, int, Scheduler[int]] = None,
                 ops: Union[None, NumpyOp, Scheduler[NumpyOp], List[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
                 num_process: Optional[int] = None):
        self.data = {x: y for (x, y) in zip(["train", "eval", "test"], [train_data, eval_data, test_data]) if y}
        self.batch_size = batch_size
        self.ops = to_list(ops)
        self.num_process = num_process if num_process is not None else os.cpu_count()
        self._verify_inputs(**{k: v for k, v in locals().items() if k != 'self'})

    def _verify_inputs(self, **kwargs):
        fe_dataset = False
        for mode, dataset in self.data.items():
            if isinstance(dataset, Scheduler):
                for ds in dataset.get_all_values():
                    fe_dataset = self._verify_dataset(mode, ds, **kwargs) or fe_dataset
            else:
                fe_dataset = self._verify_dataset(mode, dataset, **kwargs) or fe_dataset
        if not fe_dataset:
            assert kwargs['batch_size'] is None, "only support batch_size with built-in dataset in Pipeline"
            assert kwargs['ops'] is None, "only support ops with built-in dataset in Pipeline"
            assert kwargs['num_process'] is None, "only support num_process with built-in dataset in Pipeline"

    def _verify_dataset(self, mode: str, dataset: DataSource, **kwargs) -> bool:
        if isinstance(dataset, Dataset):
            # batch_size check
            assert isinstance(self.batch_size, (Scheduler, int, type(None))), \
                "unsupported batch_size format: {}".format(self.batch_size)
            if isinstance(self.batch_size, Scheduler):
                for batch_size in self.batch_size.get_all_values():
                    assert isinstance(batch_size, int, type(None)), \
                        "unsupported batch_size format: {}".format(self.batch_size)
            # ops check
            for op in self.ops:
                if isinstance(op, Scheduler):
                    for epoch_op in op.get_all_values():
                        assert isinstance(epoch_op, NumpyOp), "unsupported op format, must provide NumpyOp in Pipeline"
                else:
                    assert isinstance(op, NumpyOp), "unsupported op format, must provide NumpyOp in Pipeline"
            # num_process check
            assert isinstance(self.num_process, int), "number of processes must be an integer"
            return True
        elif isinstance(dataset, (DataLoader, tf.data.Dataset)):
            if kwargs['batch_size'] is not None:
                warnings.warn("batch_size will only be used for built-in dataset")
            if kwargs['ops'] is not None:
                warnings.warn("ops will only be used for built-in dataset")
            if kwargs['num_process'] is not None:
                warnings.warn("num_process will only be used for built-in dataset")
            return False
        else:
            raise ValueError("Unsupported dataset type for {}".format(mode))

    def get_modes(self) -> Set[str]:
        """get the active modes in pipeline

        Returns:
            set of active modes
        """
        return set(self.data.keys())

    def benchmark(self, mode: str = "train", epoch: int = 1, num_steps: int = 1000, log_interval: int = 100):
        """benchmark the pipeline processing speed

        Args:
            mode: Current mode, can be 'train', 'eval' or 'test'.
            epoch: Current epoch index. Defaults to 1.
            num_steps: Maximum number of steps to do benchmark on. Defaults to 1000.
            log_interval: Logging interval. Defaults to 100.
        """
        loader = self.get_loader(mode=mode, epoch=epoch)
        if isinstance(loader, tf.data.Dataset):
            loader = loader.take(num_steps)
        start = time.perf_counter()
        for idx, _ in enumerate(loader, start=1):
            if idx % log_interval == 0:
                duration = time.perf_counter() - start
                iters_per_sec = log_interval / duration
                print("FastEstimator: Step: {}, Epoch: {}, Steps/sec: {}".format(idx, epoch, iters_per_sec))
                start = time.perf_counter()
            if idx == num_steps:
                break

    def transform(self, data: Dict[str, Any], mode: str, epoch: int = 1) -> Dict[str, Any]:
        """apply all pipeline operations on given data for certain mode and epoch.

        Args:
            data: Input data in dictionary format
            mode: Current mode, can be "train", "eval", "test" or "infer"
            epoch: Current epoch index. Defaults to 1.

        Returns:
            transformed data
        """
        data = deepcopy(data)
        ops = get_current_ops(self.ops, mode, epoch)
        forward_numpyop(ops, data, mode)
        for key, value in data.items():
            data[key] = np.expand_dims(value, 0)
        return data

    def get_results(self, mode: str = "train", epoch: int = 1, num_steps: int = 1,
                    shuffle=False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """get the pipeline outputs after all ops

        Args:
            mode: Current mode, can be 'train', 'eval' or 'test'.
            epoch: Current epoch index. Defaults to 1.
            num_steps: number of steps(batches) to get. Defaults to 1.

        Returns:
            pipeline outputs
        """
        results = []
        loader = self.get_loader(mode=mode, epoch=epoch, shuffle=shuffle)
        if isinstance(loader, tf.data.Dataset):
            loader = loader.take(num_steps)
        for idx, batch in enumerate(loader):
            if idx == num_steps:
                break
            results.append(batch)
        if len(results) == 1:
            results = results[0]
        return results

    def get_loader(self, mode: str, epoch: int = 1,
                   shuffle: Optional[bool] = None) -> Union[DataLoader, tf.data.Dataset]:
        """get the data loader given mode and epoch

        Args:
            mode: Current mode, can be 'train', 'eval' or 'test'.
            epoch: Current epoch index. Defaults to 1.
            shuffle: Whether to shuffle, only used with FE dataset. If None, shuffle is based on mode. Defaults to None.

        Returns:
            data loader given the mode and epoch.
        """
        data = self.data[mode]
        if isinstance(data, Scheduler):
            data = data.get_current_value(epoch)
        if isinstance(data, Dataset):
            op_dataset = OpDataset(data, get_current_ops(self.ops, mode, epoch), mode)
            batch_size = self.batch_size
            if isinstance(batch_size, Scheduler):
                batch_size = batch_size.get_current_value(epoch)
            if isinstance(data, BatchDataset) and batch_size is not None:
                raise ValueError("batch_size must be None when using BatchDataset")
            if shuffle is None:
                shuffle = mode == "train"
            data = DataLoader(op_dataset,
                              batch_size=batch_size,
                              shuffle=False if isinstance(data, BatchDataset) else shuffle,
                              sampler=RandomSampler(op_dataset) if isinstance(data, BatchDataset) and shuffle else None,
                              num_workers=self.num_process,
                              worker_init_fn=lambda _: np.random.seed())
        return data

    def get_signature_epochs(self, total_epochs: int):
        """get the signature epochs that scheduler will be effective on.

        Args:
            total_epochs: total number of epochs

        Returns:
            set: set of epoch index
        """
        signature_epochs = {1}
        epoch_keys = {1}
        repeat_cycles = {1}
        for x in self.ops + list(self.data.values()) + [self.batch_size]:
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
        signature_epochs = set(epoch for epoch in signature_epochs if epoch <= total_epochs)
        return signature_epochs

    @lru_cache(maxsize=None, typed=True)
    def get_all_output_keys(self, mode: str, total_epochs: int) -> Set[str]:
        """get the pipeline output keys for a given mode

        Args:
            mode: current mode, can be "train", "eval" , "test" or "infer"
            total_epochs: total number of epochs

        Returns:
            set of all keys for given mode
        """
        output_keys = set()
        for epoch in self.get_signature_epochs(total_epochs):
            loader = self.get_loader(mode=mode, epoch=epoch)
            if isinstance(loader, DataLoader):
                if isinstance(loader.dataset, OpDataset):
                    data = loader.dataset.dataset[0]
                    for op in loader.dataset.ops:
                        output_keys.update(op.outputs)
                else:
                    data = loader.dataset[0]
            else:
                data = next(iter(loader))
            assert isinstance(data, dict), "please make sure data output format is dictionary"
            output_keys.update(data.keys())
        return output_keys
