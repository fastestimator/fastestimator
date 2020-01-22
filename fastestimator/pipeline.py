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
import sys
import time
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Set, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

from fastestimator.op import NumpyOp, get_inputs_by_op, write_outputs_by_key
from fastestimator.schedule import Scheduler
from fastestimator.util.util import to_list

T = TypeVar('T')


class GeneratorDataset(Dataset):
    def __init__(self, generator: Generator[Dict[str, Any], int, None], samples_per_epoch: int):
        self.generator = generator
        self.samples_per_epoch = samples_per_epoch
        next(self.generator)  # Can't send non-none values to a new generator, so need to run a 'warm-up' first

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        return self.generator.send(index)


class NumpyDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.size = None
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if self.size is not None:
                    assert val.shape[0] == self.size, "All data arrays must have the same number of elements"
                else:
                    self.size = val.shape[0]
        assert isinstance(self.size, int), \
            "Could not infer size of data. Please ensure you are passing numpy arrays in the data dictionary."

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {key: val[index] for key, val in self.data.items()}


class OpDataset(Dataset):
    def __init__(self, dataset: Dataset, ops: List[Union[NumpyOp, Scheduler[NumpyOp]]]):
        self.dataset = dataset
        self.signature_epochs = get_signature_epochs(ops)
        self.op_schedule = None if len(ops) == 0 else Scheduler(
            {epoch: get_per_epoch(ops, epoch)
             for epoch in self.signature_epochs})
        # The following variables could technically by computed on the fly, but don't want to waste clock cycles
        self.epoch, self.epoch_ops, self.state_dict = 0, [], []
        self.set_epoch_and_mode(0, "train")

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        op_data = None
        for op in self.epoch_ops:
            op_data = get_inputs_by_op(op, item, op_data)
            op_data = op.forward(op_data, self.state_dict)
            if op.outputs:
                write_outputs_by_key(item, op_data, op.outputs)
        return item

    def __len__(self):
        return len(self.dataset)

    def set_epoch_and_mode(self, epoch: int, mode: str):
        self.epoch = epoch
        if self.op_schedule is not None:
            self.epoch_ops = self.op_schedule.get_current_value(epoch=epoch)
            # TODO - rather than filtering every epoch the ops should probably be separated up-front
            self.epoch_ops = list(filter(lambda op: op.mode is None or op.mode == mode, self.epoch_ops))
        self.state_dict = {"epoch": epoch, "mode": mode}

    def get_signature_epochs(self) -> List[int]:
        return self.signature_epochs


class OpDataLoader(DataLoader):
    dataset: OpDataset

    def __init__(self, dataset: OpDataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
        # It may be tempting to overwrite the collate function so that it works directly with Tensorflow, but their
        # default one has some memory management tricks that are difficult to replicate, and since we also support raw
        # data loaders in the estimator we need to put the type conversion at that level anyways.
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last,
                         worker_init_fn=lambda _: np.random.seed())
        self.set_epoch_and_mode(epoch=0, mode="train")

    def set_epoch_and_mode(self, epoch: int, mode: str):
        self.dataset.set_epoch_and_mode(epoch, mode)


def get_signature_epochs(val: Union[Scheduler, Iterable[Any]]) -> List[int]:
    signature_epochs = {0}
    if isinstance(val, Scheduler):
        signature_epochs.update(val.epoch_dict.keys())
    else:
        for elem in val:
            if isinstance(elem, Scheduler):
                signature_epochs.update(elem.epoch_dict.keys())
    return sorted(signature_epochs)


def get_per_epoch(ops: Iterable[Union[T, Scheduler[T]]], epoch: int) -> List[T]:
    per_epoch = []
    for op in ops:
        if isinstance(op, Scheduler):
            per_epoch.append(op.get_current_value(epoch))
        else:
            per_epoch.append(op)
    return per_epoch


class BasePipeline:
    def __init__(self,
                 train_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset],
                 eval_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset, None]):
        self.train_data = train_data
        self.eval_data = eval_data

    def get_batch_size(self, mode: str, epoch: int) -> int:
        return None

    def get_iterator(self, mode: str, epoch: int) -> Iterator:
        itr = None
        if mode == "train":
            itr = self.train_data
        elif mode == "eval":
            itr = self.eval_data
        return itr

    def get_num_steps(self, mode: str, epoch: int) -> int:
        raise NotImplementedError()

    def index(self, idx: int, mode: str = "train", epoch: int = 0) -> dict:
        raise NotImplementedError()

    def transform(self, data: dict) -> dict:
        raise NotImplementedError()

    def benchmark(self, mode: str = "train", epoch: int = 0, num_steps: int = 1000, log_interval: int = 100):
        itr = self.get_iterator(mode=mode, epoch=epoch)
        start = time.perf_counter()
        for idx in range(num_steps + 1):
            _ = next(itr)
            if idx % log_interval == 0 and idx > 0:
                duration = time.perf_counter() - start
                batch_size = self.get_batch_size(mode=mode, epoch=epoch)
                examples_per_sec = log_interval * batch_size / duration
                print("FastEstimator: Step: {}, Epoch: {}, Batch Size: {}, Examples/sec: {}".format(
                    idx, epoch, batch_size, examples_per_sec))
                start = time.perf_counter()


class TorchPipeline(BasePipeline):
    def __init__(self, train_data, eval_data):
        super().__init__(train_data=train_data, eval_data=eval_data)

    def get_batch_size(self, mode: str, epoch: int) -> int:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        return dataloader.batch_size

    def get_num_steps(self, mode: str, epoch: int) -> int:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        return len(dataloader)

    def index(self, idx: int, mode: str = "train", epoch: int = 0) -> dict:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        dataset = dataloader.dataset
        return dataset[idx]


class TensorFlowPipeline(BasePipeline):
    def __init__(self, train_data, eval_data, batch_size):
        super().__init__(train_data=train_data, eval_data=eval_data)
        self.batch_size = batch_size

    def get_num_steps(self, mode: str, epoch: int) -> int:
        dataset = self.get_iterator(mode=mode, epoch=epoch)
        num_steps = tf.data.experimental.cardinality(dataset)
        if num_steps in [tf.data.experimental.INFINITE_CARDINALITY, tf.data.experimental.UNKNOWN_CARDINALITY]:
            raise ValueError("Cannot infer total steps, please provide 'steps_per_epoch' in Estimator")
        return int(num_steps)

    def get_batch_size(self, mode: str, epoch: int) -> int:
        if self.batch_size is None:
            raise ValueError("Cannot infer batch_size, please provide 'batch_size' in Pipeline")
        return self.batch_size


class FEPipeline(BasePipeline):
    """ A class representing the data pipeline for FastEstimator training

    Args:
        train_data (fe.data.Dataset): A fe.dataset for training.
        eval_data (fe.data.Dataset): A dataset for evaluation.
        batch_size (int, Scheduler): The batch size to use during training
        ops (list, fe.op): A list of operations to be applied within the data pipeline
        num_process (int): How many CPUs to use in the pipeline. None will auto-select based on performance tests.
                            You might need to set this to zero if you want to use debuggers
    """
    def __init__(self,
                 train_data: Union[Dataset, Scheduler],
                 eval_data: Union[Dataset, Scheduler, None] = None,
                 batch_size: Union[int, Scheduler, None] = None,
                 ops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
                 num_process: Union[int, None] = None):
        super().__init__(train_data=train_data, eval_data=eval_data)
        self.ops = [] if ops is None else to_list(ops)
        self.batch_size = batch_size
        self.num_process = os.cpu_count() if num_process is None else num_process
        self.datasets = {}
        self.dataloaders = {}
        self.datasets["train"] = self._build_dataset(train_data, "train")
        # Intentionally not using self.num_process in the build_loader call since want to trigger perf test
        self.dataloaders["train"] = self._build_loader(self.datasets["train"], shuffle=True, num_process=num_process)
        if eval_data is not None:
            self.datasets["eval"] = self._build_dataset(eval_data, "eval")
            self.dataloaders["eval"] = self._build_loader(self.datasets["eval"], shuffle=False, num_process=num_process)

    def _build_dataset(self, dataset: Dataset, mode: str) -> OpDataset:
        return OpDataset(dataset, list(filter(lambda op: op.mode in (None, mode), self.ops)))

    def _build_loader(self,
                      dataset: OpDataset,
                      shuffle: bool = True,
                      num_process: Optional[Union[int, Scheduler[int]]] = None) -> Scheduler[OpDataLoader]:
        if num_process is None:
            n_samples = 5
            cpu_list = [0, max(os.cpu_count() // 2, 1), os.cpu_count()]
            # Run some tests
            times = [
                self._test_loader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=shuffle,
                                  num_workers=i,
                                  num_samples=n_samples) for i in cpu_list
            ]
            best_idx = times.index(min(times))
            num_process = cpu_list[best_idx]
        elif num_process < 6:
            num_process = 0
        loader = OpDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_process)
        return loader

    def _test_loader(self, dataset: OpDataset, batch_size: int, shuffle: bool, num_workers: int,
                     num_samples: int) -> float:
        loader = OpDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        itr = iter(loader)
        # warmup
        _ = next(itr)
        # time a few iterations to find out which method is best
        start = time.perf_counter()
        for i in range(num_samples):
            _ = next(itr)
        return time.perf_counter() - start

    def get_iterator(self, mode: str, epoch: int) -> Iterator:
        return self.dataloaders[mode]

    def get_batch_size(self, mode: str, epoch: int) -> int:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        return dataloader.batch_size

    def get_num_steps(self, mode: str, epoch: int) -> int:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        return len(dataloader)

    def index(self, idx: int, mode: str = "train", epoch: int = 0) -> dict:
        dataloader = self.get_iterator(mode=mode, epoch=epoch)
        dataset = dataloader.dataset
        return dataset[idx]


def Pipeline(train_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset],
             eval_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset, None] = None,
             batch_size: Union[int, Scheduler, None] = None,
             ops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
             num_process: Union[int, None] = None):
    assert isinstance(train_data, (Dataset, DataLoader, Scheduler, tf.data.Dataset))
    if not eval_data:
        assert type(train_data) == type(eval_data)
    if isinstance(train_data, tf.data.Dataset):
        pipeline = TensorFlowPipeline(train_data, eval_data, batch_size)
    elif isinstance(train_data, DataLoader):
        pipeline = TorchPipeline(train_data, eval_data)
    else:
        pipeline = FEPipeline(train_data, eval_data, batch_size, ops=ops, num_process=num_process)
    return pipeline
