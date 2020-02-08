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
from typing import List, Set, Union

import tensorflow as tf
from torch.utils.data import DataLoader, Dataset

from fastestimator.dataset.fe_dataset import OpDataset
from fastestimator.op import NumpyOp, get_current_ops
from fastestimator.schedule import EpochScheduler, RepeatScheduler, Scheduler
from fastestimator.util.util import lcms, to_list


class Pipeline:
    def __init__(self,
                 train_data: Union[Dataset, DataLoader, Scheduler, tf.data.Dataset, None] = None,
                 eval_data: Union[Dataset, DataLoader, tf.data.Dataset, None] = None,
                 test_data: Union[Dataset, DataLoader, tf.data.Dataset, None] = None,
                 batch_size: Union[int, Scheduler, None] = None,
                 ops: Union[None, NumpyOp, Scheduler, List[Union[NumpyOp, Scheduler]]] = None,
                 num_process: Union[int, None] = None):
        self.data = {x: y for (x, y) in zip(["train", "eval", "test"], [train_data, eval_data, test_data]) if y}
        self.batch_size = batch_size
        self.ops = ops
        self.num_process = num_process
        self.fe_dataset_exist = False
        self._verify_inputs()
        self.ops = [] if ops is None else to_list(ops)
        self.num_process = os.cpu_count() if num_process is None else num_process

    def _verify_inputs(self):
        for mode, dataset in self.data.items():
            if isinstance(dataset, (RepeatScheduler, EpochScheduler)):
                for ds in dataset.get_items():
                    self._verify_dataset(mode, ds)
            else:
                self._verify_dataset(mode, dataset)
        if not self.fe_dataset_exist:
            assert not self.batch_size, "only support batch_size with built-in dataset in Pipeline"
            assert not self.ops, "only support ops with built-in dataset in Pipeline"
            assert not self.num_process, "only support num_process with built-in dataset in Pipeline"

    def _verify_dataset(self, mode, dataset):
        if isinstance(dataset, Dataset):
            self.fe_dataset_exist = True
            #batch_size check
            assert isinstance(self.batch_size, (RepeatScheduler, EpochScheduler, int)), "unsupported batch_size format"
            if isinstance(self.batch_size, Scheduler):
                for batch_size in self.batch_size.get_items():
                    assert isinstance(batch_size, int), "unsupported batch_size format"
            #ops check
            assert isinstance(self.ops, list), "unsupported ops format"
            for op in self.ops:
                if isinstance(op, (RepeatScheduler, EpochScheduler)):
                    for epoch_op in op.get_items():
                        assert isinstance(epoch_op, NumpyOp), "unsupported op format, must provide NumpyOp in Pipeline"
                else:
                    assert isinstance(op, NumpyOp), "unsupported op format, must provide NumpyOp in Pipeline"
            #num_process check
            if self.num_process:
                assert isinstance(self.num_process, int), "number of process must be integer or None"
        elif isinstance(dataset, (DataLoader, tf.data.Dataset)):
            if self.batch_size:
                warnings.warn("batch_size will only be used for built-in dataset")
            if self.ops:
                warnings.warn("ops will only be used for built-in dataset")
            if self.num_process:
                warnings.warn("num_process will only be used for built-in dataset")
        else:
            raise ValueError("Unsupported dataset type for {}".format(mode))

    def get_modes(self) -> Set[str]:
        return set(mode for mode in self.data.keys())

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

    def get_loader(self, mode: str, epoch: int = 0):
        data = self.data[mode]
        if isinstance(data, Scheduler):
            data = data.get_current_value(epoch)
        if isinstance(data, Dataset):
            op_dataset = OpDataset(data, get_current_ops(self.ops, mode, epoch), mode)
            batch_size = self.batch_size
            if isinstance(batch_size, Scheduler):
                batch_size = batch_size.get_current_value(epoch)
            data = DataLoader(op_dataset, batch_size=batch_size, shuffle=mode == "train", num_workers=self.num_process)
        return data

    def get_signature_epoches(self, total_epoches):
        signature_epoches = {0}
        epoch_keys = {0}
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
                signature_epoches.update(range(epoch, epoch + min(epoch_keys[idx + 1] - epoch, least_common_cycle)))
            else:
                signature_epoches.update(range(epoch, epoch + least_common_cycle))
        signature_epoches = set(epoch for epoch in signature_epoches if epoch < total_epoches)
        return signature_epoches

    def get_all_output_keys(self, mode, total_epoches) -> Set[str]:
        output_keys = set()
        for epoch in self.get_signature_epoches(total_epoches):
            loader = self.get_loader(mode=mode, epoch=epoch)
            data = next(iter(loader))
            assert isinstance(data, dict), "please make sure data output format is dictionary"
            output_keys = output_keys.union(set(data.keys()))
            if isinstance(loader, DataLoader) and isinstance(loader.dataset, OpDataset):
                for op in loader.dataset.ops:
                    output_keys.update(to_list(op.outputs))
        output_keys -= {None}
        return output_keys
