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
from typing import Optional, Union, Iterable, Iterator, Any, List, TypeVar, Dict, Set, Generator

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

from fastestimator.op import get_inputs_by_op, write_outputs_by_key, NumpyOp
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

    def __init__(self, dataset: OpDataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
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
    def get_modes(self) -> Set[str]:
        raise NotImplementedError()

    def get_global_batch_size(self, mode: str, epoch: int) -> int:
        raise NotImplementedError()

    def get_signature_epochs(self, mode: str) -> List[int]:
        raise NotImplementedError()

    def get_all_output_keys(self) -> Set[str]:
        raise NotImplementedError()

    def get_num_examples(self, mode: str, epoch: int) -> int:
        raise NotImplementedError()

    def transform(self, mode: str, epoch: int = 0, dataset: Optional[Dataset] = None) -> Iterator:
        raise NotImplementedError()

    def benchmark(self, mode: str = "train", num_steps: int = 1000, log_interval: int = 100, epoch: int = 0):
        itr = self.transform(mode=mode, epoch=epoch)
        start = time.perf_counter()
        for idx in range(num_steps + 1):
            _ = next(itr)
            if idx % log_interval == 0 and idx > 0:
                duration = time.perf_counter() - start
                examples_per_sec = log_interval * self.get_global_batch_size(mode=mode, epoch=epoch) / duration
                print("FastEstimator: Step: {}, Epoch: {}, Batch Size: {}, Examples/sec: {}".format(
                    idx, epoch, self.get_global_batch_size(mode=mode, epoch=epoch), examples_per_sec))
                start = time.perf_counter()

    def show_results(self, mode: str = "train", num_steps: int = 1, epoch: int = 0):
        data = []
        itr = self.transform(mode=mode, epoch=epoch)
        for _ in range(num_steps):
            data.append(next(itr))
        return data


class Pipeline(BasePipeline):
    # TODO support filter ops
    # TODO support cache op
    dataloaders: Dict[str, Scheduler[DataLoader]]
    batch_size: Scheduler[int]
    """ A class representing the data pipeline for FastEstimator

    Args:
        train_data (torch.utils.data.Dataset): A dataset for training. Required for training
        eval_data (torch.utils.data.Dataset): A dataset for evaluation
        batch_size (int, Scheduler): The batch size to use during training
        ops (list, fe.op): A list of operations to be applied within the data pipeline
        drop_last (bool): Whether to drop the last batch in an epoch if there aren't enough remaining elements for a
                        complete batch
        num_process (int): How many CPUs to use in the pipeline. None will auto-select based on performance tests. 
                            You might need to set this to zero if you want to use debuggers
        shuffle_train (bool): Whether to shuffle the training data
        shuffle_eval (bool): Whether to shuffle the eval data
    """
    def __init__(self,
                 train_data: Optional[Dataset] = None,
                 eval_data: Optional[Dataset] = None,
                 batch_size: Union[int, Scheduler] = 1,
                 ops: Union[None, NumpyOp, Scheduler[NumpyOp], Iterable[Union[NumpyOp, Scheduler[NumpyOp]]]] = None,
                 drop_last: bool = False,
                 num_process: Optional[Union[int, Scheduler[int]]] = None,
                 shuffle_train: bool = True,
                 shuffle_eval: bool = False):
        self.ops = [] if ops is None else to_list(ops)
        if isinstance(batch_size, Scheduler):
            assert 0 in batch_size.epoch_dict.keys(), "Batch size must be specified for epoch 0"
            assert all(map(lambda x: x is not None, batch_size.epoch_dict.values())), "Batch size must never be None"
        else:
            batch_size = Scheduler({0: batch_size})
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_process = os.cpu_count() or 1 if num_process is None else num_process
        self.datasets = {}
        self.dataloaders = {}
        if train_data is not None:
            self.datasets["train"] = self._build_dataset(train_data, "train")
            self.dataloaders["train"] = self._build_loader(self.datasets["train"], shuffle=shuffle_train)
        if eval_data is not None:
            self.datasets["eval"] = self._build_dataset(eval_data, "eval")
            self.dataloaders["eval"] = self._build_loader(self.datasets["eval"], shuffle=shuffle_eval)
        # All output keys from the dataset, plus any outputs from the ops
        self.all_output_keys = {key for dataset in self.datasets.values() for key in dataset[0].keys()}
        for op in self.ops:
            if isinstance(op, Scheduler):
                self.all_output_keys.update(map(lambda x: to_list(x.outputs), op.epoch_dict.values()))
            else:
                self.all_output_keys.update(to_list(op.outputs))
        self.all_output_keys -= {None}

    def _build_dataset(self, dataset: Dataset, mode: str) -> OpDataset:
        return OpDataset(dataset, list(filter(lambda op: op.mode in (None, mode), self.ops)))

    def _build_loader(self,
                      dataset: OpDataset,
                      shuffle: bool = True,
                      num_process: Optional[Union[int, Scheduler[int]]] = None) -> Scheduler[OpDataLoader]:
        # TODO - technically the signature epochs here shouldn't be based only on batch size, but also any keys within
        #  num_process as well as the signature epochs of the ops
        if num_process is not None:
            if isinstance(num_process, int):
                num_process = Scheduler({0: num_process})
            return Scheduler({
                epoch: OpDataLoader(dataset,
                                    batch_size=size,
                                    shuffle=shuffle,
                                    num_workers=num_process.get_current_value(epoch),
                                    drop_last=self.drop_last)
                for epoch,
                size in self.batch_size.epoch_dict.items()
            })
        # We're going to test whether single process or multi-process is faster for this problem. Multi-processing has
        # been observed to be dramatically worse than single process for problems with inexpensive pipelines and low
        # batch sizes
        epoch_dict = {}
        for epoch, size in self.batch_size.epoch_dict.items():
            n_samples = 5
            if len(dataset) < n_samples + 1:
                # If there aren't enough samples to perform a comparison then just use single processing since there's
                # not enough work for multiple cores anyways
                epoch_dict[epoch] = OpDataLoader(dataset,
                                                 batch_size=size,
                                                 shuffle=shuffle,
                                                 num_workers=0,
                                                 drop_last=self.drop_last)
                continue
            # Run some tests
            times = [
                self._test_loader(dataset=dataset,
                                  batch_size=size,
                                  shuffle=shuffle,
                                  num_workers=i,
                                  num_samples=n_samples) for i in [0, max(os.cpu_count() // 2, 1), os.cpu_count()]
            ]
            best = times.index(min(times))
            if best == 0:
                epoch_dict[epoch] = OpDataLoader(dataset,
                                                 batch_size=size,
                                                 shuffle=shuffle,
                                                 num_workers=0,
                                                 drop_last=self.drop_last)
            elif best == 1:
                epoch_dict[epoch] = OpDataLoader(dataset,
                                                 batch_size=size,
                                                 shuffle=shuffle,
                                                 num_workers=max(os.cpu_count() // 2, 1),
                                                 drop_last=self.drop_last)
            elif best == 2:
                epoch_dict[epoch] = OpDataLoader(dataset,
                                                 batch_size=size,
                                                 shuffle=shuffle,
                                                 num_workers=os.cpu_count(),
                                                 drop_last=self.drop_last)
        return Scheduler(epoch_dict=epoch_dict)

    def _test_loader(self, dataset: OpDataset, batch_size: int, shuffle: bool, num_workers: int,
                     num_samples: int) -> float:
        loader = OpDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=self.drop_last)
        itr = iter(loader)
        # warmup
        _ = next(itr)
        # time a few iterations to find out which method is best
        start = time.perf_counter()
        for i in range(num_samples):
            _ = next(itr)
        return time.perf_counter() - start

    def get_modes(self) -> Set[str]:
        return set(self.dataloaders.keys())

    def add_dataset(self,
                    dataset: Dataset,
                    mode: str,
                    shuffle: bool = True,
                    num_process: Optional[Union[int, Scheduler[int]]] = None):
        self.datasets[mode] = self._build_dataset(dataset, mode)
        self.dataloaders[mode] = self._build_loader(self.datasets[mode], shuffle=shuffle, num_process=num_process)

    def get_global_batch_size(self, mode: str, epoch: int) -> int:
        # TODO - figure out global vs local. I'm pretty sure this loader will always be global batch
        # TODO - support per-mode batch size
        return self.batch_size.get_current_value(epoch)

    def get_num_examples(self, mode: str, epoch: int) -> int:
        if self.dataloaders.get(mode) is None:
            return 0
        return len(self.dataloaders[mode].get_current_value(epoch).dataset)

    def get_signature_epochs(self, mode: str) -> List[int]:
        sig_op_epochs = {} if self.datasets.get(mode) is None else set(self.datasets[mode].get_signature_epochs())
        return sorted(self.batch_size.epoch_dict.keys() | sig_op_epochs)

    def get_all_output_keys(self) -> Set[str]:
        return self.all_output_keys

    def transform(self, mode: str, epoch: int = 0, dataset: Optional[Dataset] = None) -> Iterator:
        if dataset is None:
            loader = self.dataloaders.get(mode)
            if loader is not None:
                loader = loader.get_current_value(epoch)
        else:
            if not isinstance(dataset, OpDataset):
                dataset = self._build_dataset(dataset, mode)
            loader = self._build_loader(dataset, shuffle=False).get_current_value(epoch)
        if loader is None:
            return iter(())
        loader.set_epoch_and_mode(epoch, mode)
        return iter(loader)


class TorchPipeline(BasePipeline):
    def __init__(self, dataloaders: Dict[str, DataLoader]):
        self.dataloaders = dataloaders
        self.num_examples = {mode: len(loader.dataset) for mode, loader in dataloaders.items()}
        self.batch_sizes = {mode: loader.batch_size for mode, loader in dataloaders.items()}
        self.output_keys = set()
        for mode, loader in self.dataloaders.items():
            batch = next(iter(loader))
            assert isinstance(batch, dict), "DataLoader must return a dictionary of values"
            for key, value in batch.items():
                self.output_keys.add(key)
                # If the user had a custom batch sampler then the loader's batch size will be None. Need to infer it
                if self.batch_sizes[mode] is None:
                    self.batch_sizes[mode] = _infer_batch_size(value)

    def get_modes(self) -> Set[str]:
        return set(self.dataloaders.keys())

    def get_num_examples(self, mode: str, epoch: int) -> int:
        return self.num_examples.get(mode, 0)

    def get_signature_epochs(self, mode: str) -> List[int]:
        return [0]

    def get_global_batch_size(self, mode: str, epoch: int) -> int:
        return self.batch_sizes.get(mode, 0)

    def get_all_output_keys(self) -> Set[str]:
        return self.output_keys

    def transform(self, mode: str, epoch: int = 0, dataset: Optional[Dataset] = None) -> Iterator:
        if dataset is not None:
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_sizes.get(mode, 1),
                                num_workers=os.cpu_count(),
                                drop_last=False)
        else:
            loader = self.dataloaders.get(mode, ())
        return iter(loader)


class TensorFlowPipeline(BasePipeline):
    def __init__(self, dataloaders: Dict[str, tf.data.Dataset]):
        self.dataloaders = dataloaders
        self.batch_sizes = {mode: None for mode in dataloaders.keys()}
        self.num_examples = {mode: tf.data.experimental.cardinality(dataset) for mode, dataset in dataloaders.items()}
        self.output_keys = set()
        for mode, loader in self.dataloaders.items():
            batch = next(iter(loader))
            assert isinstance(batch, dict), "Dataset must return a dictionary of values"
            for key, value in batch.items():
                self.output_keys.add(key)
                # If the user had a custom batch sampler then the loader's batch size will be None. Need to infer it
                if self.batch_sizes[mode] is None:
                    self.batch_sizes[mode] = _infer_batch_size(value)
        for mode, size in self.num_examples.items():
            if size == tf.data.experimental.INFINITE_CARDINALITY:
                self.num_examples[mode] = sys.maxsize  # TODO - handle infinity in a clever way somehow
            if size == tf.data.experimental.UNKNOWN_CARDINALITY:
                self.num_examples[mode] = self.dataloaders[mode].reduce(0, lambda x, _: x + 1)
            else:
                # Cardinality is based on batch size, but we want total number of examples
                self.num_examples[mode] = int(self.num_examples[mode]) * self.batch_sizes[mode]

    def get_modes(self) -> Set[str]:
        return set(self.dataloaders.keys())

    def get_global_batch_size(self, mode: str, epoch: int) -> int:
        return self.batch_sizes.get(mode, 0)

    def get_signature_epochs(self, mode: str) -> List[int]:
        return [0]

    def get_all_output_keys(self) -> Set[str]:
        return self.output_keys

    def get_num_examples(self, mode: str, epoch: int) -> int:
        return self.num_examples.get(mode, 0)

    def transform(self, mode: str, epoch: int = 0, dataset: Optional[Dataset] = None) -> Iterator:
        if dataset is not None:
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch_sizes.get(mode, 1),
                                num_workers=os.cpu_count(),
                                drop_last=False)
        else:
            loader = self.dataloaders.get(mode, ())
        return iter(loader)


def torch_to_tf(data):
    # TODO - it might be desirable to replace the collate function of the data loader rather than casting
    #  after-the-fact, but surprisingly tests so far have shown that this doesn't add a noticeable performance penalty
    if isinstance(data, tf.Tensor):
        return data
    if isinstance(data, torch.Tensor):
        return tf.constant(data.numpy(), dtype=tf.float32)
    if isinstance(data, dict):
        result = {}
        for key, val in data.items():
            result[key] = torch_to_tf(val)
        return result
    if isinstance(data, list):
        return [torch_to_tf(val) for val in data]
    if isinstance(data, tuple):
        return tuple([torch_to_tf(val) for val in data])
    if isinstance(data, set):
        return set([torch_to_tf(val) for val in data])


def _infer_batch_size(data) -> int:
    if isinstance(data, (torch.Tensor, tf.Tensor)):
        return data.shape[0]
    if isinstance(data, (list, tuple)):
        return _infer_batch_size(data[0])
    if isinstance(data, set):
        return _infer_batch_size(data.pop())
    if isinstance(data, dict):
        return _infer_batch_size(data.popitem()[1])
    raise ValueError("Could not infer batch size from data loader")
