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
from typing import Any, Dict, Iterable, List, Optional, Set, TypeVar, Union

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

from fastestimator.op import NumpyOp, get_inputs_by_op, get_ops_by_mode, write_outputs_by_key
from fastestimator.schedule import Scheduler
from fastestimator.util.util import to_list

T = TypeVar('T')


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
        self.effective_outputs = set()

    def __getitem__(self, index):
        item = self.dataset[index]
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


class TorchPipeline:
    def __init__(self, dataloaders: dict):
        self.dataloaders = dataloaders

    def get_num_steps(self, mode: str, epoch: int) -> int:
        loader = self.get_loader(mode=mode, epoch=epoch)
        return len(loader)

    def get_modes(self) -> Set[str]:
        return set([mode for mode in self.dataloaders.keys() if self.dataloaders[mode] is not None])

    def get_loader(self, mode: str, epoch: int = 0):
        return self.dataloaders[mode]

    def benchmark(self, mode: str = "train", num_steps: int = 1000, log_interval: int = 100, epoch: int = 0):
        loader = self.get_loader(mode=mode, epoch=epoch)
        start = time.perf_counter()
        for idx, _ in enumerate(loader):
            if idx % log_interval == 0 and idx > 0:
                duration = time.perf_counter() - start
                iters_per_sec = log_interval / duration
                print("FastEstimator: Step: {}, Epoch: {}, Steps/sec: {}".format(idx, epoch, iters_per_sec))
                start = time.perf_counter()
            if idx == num_steps:
                break

    def get_all_output_keys(self) -> Set[str]:
        modes = self.get_modes()
        output_keys = set()
        for mode in modes:
            itr = iter(self.get_loader(mode=mode, epoch=0))
            data = next(itr)
            assert isinstance(data, dict), "please make sure data output format is dictionary"
            output_keys = output_keys.union(set(data.keys()))
        return output_keys


class FEPipeline(TorchPipeline):
    dataloaders: Dict[str, Scheduler[DataLoader]]
    batch_size: Scheduler[int]
    """ A class representing the data pipeline for FastEstimator
    Args:
        train_data (torch.utils.data.Dataset): A dataset for training. Required for training
        eval_data (torch.utils.data.Dataset): A dataset for evaluation
        batch_size (int, Scheduler): The batch size to use during training
        ops (list, fe.op): A list of operations to be applied within the data pipeline
        num_process (int): How many CPUs to use in the pipeline. None will auto-select based on performance tests.
                            You might need to set this to zero if you want to use debuggers
    """
    def __init__(self,
                 train_data: Optional[Dataset] = None,
                 eval_data: Optional[Dataset] = None,
                 batch_size: Union[int, Scheduler] = 1,
                 ops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
                 num_process: Optional[Union[int, Scheduler[int]]] = None):
        self.batch_size = batch_size
        self.ops = [] if ops is None else to_list(ops)
        self.num_process = os.cpu_count() if num_process is None else num_process
        self.datasets = {}
        dataloaders = {}
        if train_data is not None:
            self.datasets["train"] = self._build_dataset(train_data, "train")
            dataloaders["train"] = self._build_loader(self.datasets["train"], shuffle=True)
        if eval_data is not None:
            self.datasets["eval"] = self._build_dataset(eval_data, "eval")
            dataloaders["eval"] = self._build_loader(self.datasets["eval"], shuffle=False)
        super().__init__(dataloaders=dataloaders)

    def _build_dataset(self, dataset: Dataset, mode: str) -> OpDataset:
        return OpDataset(dataset, get_ops_by_mode(self.ops, mode))

    def _build_loader(self, dataset: OpDataset, shuffle: bool = True):
        return OpDataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_process)

    def get_all_output_keys(self) -> Set[str]:
        modes = self.get_modes()
        output_keys = set()
        for mode in modes:
            itr = iter(self.get_loader(mode=mode, epoch=0).dataset.dataset)
            data = next(itr)
            assert isinstance(data, dict), "please make sure data output format is dictionary"
            output_keys = output_keys.union(set(data.keys()))
            for op in self.ops:
                if isinstance(op, Scheduler):
                    keys_lists = [to_list(x.outputs) for x in op.epoch_dict.values() if not x is None]
                    for keys_list in keys_lists:
                        output_keys.update(keys_list)
                else:
                    output_keys.update(to_list(op.outputs))
        output_keys -= {None}
        return output_keys


class TFPipeline(TorchPipeline):
    def get_num_steps(self, mode: str, epoch: int) -> int:
        dataset = self.get_loader(mode=mode, epoch=epoch)
        num_steps = int(tf.data.experimental.cardinality(dataset))
        if num_steps < 1:
            raise ValueError("Cannot infer total steps, please provide 'steps_per_epoch' in Estimator")
        return num_steps


# noinspection PyPep8Naming
def Pipeline(train_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset, None] = None,
             eval_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset, None] = None,
             batch_size: Union[int, Scheduler] = 1,
             ops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
             num_process: Union[int, None] = None) -> TorchPipeline:
    data = train_data if train_data is not None else eval_data
    assert data is not None, "At least one of train_data or eval_data must be provided"
    if train_data is not None and eval_data is not None:
        if isinstance(train_data, (DataLoader, tf.data.Dataset)) or isinstance(eval_data,
                                                                               (DataLoader, tf.data.Dataset)):
            assert type(train_data) == type(eval_data), "Cannot mix and match Torch Loaders and tf Datasets"
    if isinstance(data, tf.data.Dataset):
        pipeline = TFPipeline({"train": train_data, "eval": eval_data})
    elif isinstance(data, DataLoader):
        pipeline = TorchPipeline({"train": train_data, "eval": eval_data})
    else:
        pipeline = FEPipeline(train_data, eval_data, batch_size, ops=ops, num_process=num_process)
    return pipeline
