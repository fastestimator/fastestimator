# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import random
from abc import ABC
from typing import Any, Optional, Callable, Sized, Dict, List, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter

from fastestimator.dataset.extend_dataset import ExtendDataset
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.util.data import FilteredData


class FEDataLoader(DataLoader):
    """A Data Loader that can handle filtering data.

    This class is intentionally not @traceable.

    Args:
        dataset: The dataset to be drawn from. The dataset may optionally implement .fe_reset_ds(bool) and/or
            .fe_batch_indices(int) methods to modify the system's sampling behavior. See fe.dataset.BatchDataset for an
            example which uses both of these methods.
        batch_size: The batch size to use (or None if the dataset is already providing a batch).
        steps_per_epoch: How many steps to have per epoch. If None the loader will perform a single pass through the
            dataset (unless samples are filtered with replacement, in which case the dataset may be passed over multiple
            times). If `steps_per_epoch` is set, it will truncate or expand the dataset until the specified number of
            steps are reached. When expanding datasets, they will be exhausted in their entirety before being
            re-sampled, equivalent to running multiple epochs of training one after the other (unless you are also
            filtering data, in which case at most one batch of data might be seen after the re-shuffling occurs).
        shuffle: Whether to shuffle the dataset.
        num_workers: How many multiprocessing threads to use (unix/mac only).
        collate_fn: What function to use to collate a list of data into a batch. This should take care of any desired
            padding.
        drop_last: Whether to drop the last batch of data if that batch is incomplete. Note that this is meaningless for
            batched datasets, as well as when `steps_per_epoch` is set - in which case the dataset will be re-sampled as
            necessary until the specified number of steps has been completed in full.
    """
    _current_threads = []
    # The typing for 'dataset' should be an 'and' rather than 'or' but that feature is still under development:
    # https://github.com/python/typing/issues/213

    def __init__(self,
                 dataset: Union[Dataset, Sized],
                 batch_size: Optional[int] = 1,
                 steps_per_epoch: Optional[int] = None,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 collate_fn: Callable = None,
                 drop_last: bool = False):
        reset_fn = dataset.fe_reset_ds if hasattr(dataset, 'fe_reset_ds') else None
        convert_fn = dataset.fe_batch_indices if hasattr(dataset, 'fe_batch_indices') else None
        sampler = InfiniteSampler(data_source=dataset, shuffle=shuffle, reset_fn=reset_fn, convert_fn=convert_fn)
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be None or a positive integer, but got {batch_size}")
        # Figure out the real batch size. This is already done in OpDataset, but if user manually instantiates this
        # loader without using an OpDataset we still want to know the batch size
        if not hasattr(dataset, "fe_batch"):
            sample_item = dataset[0]
            dataset.fe_batch = len(sample_item) if isinstance(sample_item, list) else 0
        if dataset.fe_batch:
            # The batch size where torch is concerned is probably None, but we know that it is secretly batched
            self.fe_batch_size = dataset.fe_batch
        else:
            self.fe_batch_size = batch_size
        # Figure out how many samples should be returned during the course of 1 epoch
        if steps_per_epoch is not None:
            to_yield = steps_per_epoch * (batch_size or 1)
            # Note that drop_last is meaningless here since we will provide exactly the requested number of steps
        else:
            if isinstance(dataset, OpDataset) and isinstance(dataset.dataset, ExtendDataset):
                to_yield = dataset.dataset.spoof_length
            elif isinstance(dataset, ExtendDataset):
                to_yield = dataset.spoof_length
            else:
                to_yield = len(dataset)
            if drop_last:
                to_yield -= to_yield % (batch_size or 1)
        self.fe_samples_to_yield = to_yield
        self.fe_drop_last = drop_last
        if collate_fn is None:
            if self.fe_batch_size is not None:
                # We use the 'real' batch size here to ensure that batch datasets get collated as expected
                collate_fn = default_collate
            else:
                collate_fn = default_convert
        self.fe_collate_fn = collate_fn

        # We could disable pre-collating when num_workers=0, but this would lead to inconsistent batch ordering between
        # single- and multi-processing.

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=num_workers,
                         persistent_workers=False,
                         collate_fn=functools.partial(_pre_collate, try_fn=collate_fn),
                         worker_init_fn=lambda _: np.random.seed(random.randint(0, 2 ** 32 - 1)))

    def shutdown(self) -> None:
        """Close the worker threads used by this iterator.

        The hope is that this will prevent "RuntimeError: DataLoader worker (pid(s) XXXX) exited unexpectedly" during
        the test suites.
        """
        if isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            self._iterator._shutdown_workers()
        self._iterator = None
        FEDataLoader._current_threads.clear()

    def __iter__(self) -> _BaseDataLoaderIter:
        # Similar to the original iter method, but we remember iterators in order to manually close them when new ones
        # are created
        self.shutdown()
        self._iterator = self._get_fe_iterator()
        if isinstance(self._iterator, _MultiProcessingDataLoaderIter):
            FEDataLoader._current_threads.extend([w.pid for w in self._iterator._workers])
        return self._iterator

    def _get_fe_iterator(self):
        if self.num_workers == 0:
            if self.batch_size is None:
                # We use 'fake' batch size here to identify datasets which perform their own batching
                return _SPPostBatchIter(self)
            return _SPPreBatchIter(self)
        else:
            if self.batch_size is None:
                # We use 'fake' batch size here to identify datasets which perform their own batching
                return _MPPostBatchIter(self)
            return _MPPreBatchIter(self)


def _pre_collate(data: List[Union[FilteredData, Dict[str, Any]]], try_fn: Callable) -> \
        Tuple[bool, Union[Dict[str, Any], List[Union[FilteredData, Dict[str, Any]]]]]:
    """A function which will try to pre-collate the data ahead of time in order to accelerate 'happy path'

    Args:
        data: An un-batched batch of data
        try_fn: A collate function to attempt to use on the data

    Returns:
        The collated data (if the `try_fn` succeeded), along with the original uncollated data.
    """
    try:
        collated = try_fn(data)
        return True, collated
    except:
        # The presence of filtered data instances could break the possibly-user-specified collate function for any
        # reason, so cast a broad net on this except clause.
        return False, data


class _BaseFELoaderIter(_BaseDataLoaderIter, ABC):
    """A base class for the FE Data iterators.

    Args:
        loader: The parent loader object that will own this iterator.
    """
    def __init__(self, loader: FEDataLoader):
        super().__init__(loader)
        self.fe_batch_size = loader.fe_batch_size
        self.fe_drop_last = loader.fe_drop_last
        self.fe_collate_fn = loader.fe_collate_fn
        self.fe_samples_to_yield = loader.fe_samples_to_yield
        self.fe_samples_yielded = 0
        self.fe_extra_data = []


def _next_pre_batch(self: _BaseFELoaderIter) -> Dict[str, Tensor]:
    """The __next__ function to use for a loader which is loading individual data instances (not a batchDataset)

    There is a failure mode with this logic where if the user artificially expands their dataset while using a filter,
    and if filtered data is extremely rare (like 1 data point per epoch), and that filtered point happens to get put in
    the same batch as point X multiple times, then the final batch of data may contain multiple copies of X.

    Args:
        self: The loader iterator onto which this method is grafted.

    Returns:
        A batch of data, collated into pytorch tensors.
    """
    if self._sampler_iter is None:
        self._reset()  # This copies the pytorch implementation, and seems to work, but is clearly missing an argument
    if self.fe_samples_yielded >= self.fe_samples_to_yield:
        self.fe_extra_data.clear()  # Throw out any old data
        raise StopIteration
    # Look for new data until you have enough to complete a batch
    while len(self.fe_extra_data) < self.fe_batch_size and self.fe_samples_yielded + len(
            self.fe_extra_data) < self.fe_samples_to_yield:
        collated, candidate_batch = self._next_data()
        if collated:
            if self.fe_samples_yielded + self.fe_batch_size + len(self.fe_extra_data) < self.fe_samples_to_yield:
                # This batch can be used as-is
                self.fe_samples_yielded += self.fe_batch_size
                return candidate_batch
            # We are at the last batch of data for the epoch
            if not self.fe_extra_data and self.fe_samples_yielded + self.fe_batch_size == self.fe_samples_to_yield:
                # If we have no left-over data and the current batch is exactly the right size, then just use it
                self.fe_samples_yielded += self.fe_batch_size
                return candidate_batch
            # Otherwise we need to unpack this batch, combine it with any older data, and shrink it to the desired size
            try:
                candidate_batch = decollate_batch(batch=candidate_batch)
            except TypeError:
                print("\033[93m {}\033[00m".format("FastEstimator-Warn: One or more data points may be missing from "
                                                   "the final batch in your epoch. Please override the FE "
                                                   "decollate_batch method or share your collate_fn with the FE "
                                                   "developers for assistance."))
                break
        for instance in candidate_batch:
            if self.fe_samples_yielded + len(self.fe_extra_data) == self.fe_samples_to_yield:
                # Avoid processing extra data (needed for edge cases involving lots of filtered data near end of epoch)
                break
            if isinstance(instance, FilteredData):
                if not instance.replacement:
                    self.fe_samples_yielded += 1
                continue
            self.fe_extra_data.append(instance)
    n_keep = min(self.fe_batch_size, self.fe_samples_to_yield - self.fe_samples_yielded)
    real_batch = self.fe_extra_data[:n_keep]
    self.fe_extra_data = self.fe_extra_data[n_keep:]
    self.fe_samples_yielded += len(real_batch)
    # Enforce drop_last if required
    if not real_batch or (self.fe_drop_last and len(real_batch) < self.fe_batch_size):
        self.fe_extra_data.clear()  # Throw out any extra data
        raise StopIteration
    return self.fe_collate_fn(real_batch)


def _next_post_batch(self: _BaseFELoaderIter) -> Dict[str, Tensor]:
    """The __next__ function to use for a loader which is loading batched data (ex. for a BatchDataset)

    Args:
        self: The loader iterator onto which this method is grafted.

    Returns:
        A batch of data, collated into pytorch tensors.
    """
    if self._sampler_iter is None:
        self._reset()  # This copies the pytorch implementation, and seems to work, but is clearly missing an argument
    while self.fe_samples_yielded < self.fe_samples_to_yield:
        collated, candidate_batch = self._next_data()
        if collated:
            self.fe_samples_yielded += 1
            return candidate_batch
        for instance in candidate_batch:
            if isinstance(instance, FilteredData):
                if not instance.replacement:
                    self.fe_samples_yielded += 1
                break
        else:
            # The else block is reached iff the for loop never breaks
            self.fe_samples_yielded += 1
            return self.fe_collate_fn(candidate_batch)
    raise StopIteration


class _SPPreBatchIter(_BaseFELoaderIter, _SingleProcessDataLoaderIter):
    __next__ = _next_pre_batch


class _SPPostBatchIter(_BaseFELoaderIter, _SingleProcessDataLoaderIter):
    __next__ = _next_post_batch


class _MPPreBatchIter(_BaseFELoaderIter, _MultiProcessingDataLoaderIter):
    __next__ = _next_pre_batch


class _MPPostBatchIter(_BaseFELoaderIter, _MultiProcessingDataLoaderIter):
    __next__ = _next_post_batch


class InfiniteSampler(Sampler):
    """A class which never stops sampling.

    Args:
        data_source: The dataset to be sampled.
        shuffle: Whether to shuffle when sampling.
        reset_fn: A function to be invoked (using the provided `shuffle` arg) every time the dataset has been fully
            traversed.
        convert_fn: A function to be invoked (using the current index) every sample in order to convert an integer index
            into some arbitrary alternative index representation.
    """
    def __init__(self,
                 data_source: Sized,
                 shuffle: bool = True,
                 reset_fn: Optional[Callable[[bool], None]] = None,
                 convert_fn: Optional[Callable[[int], Any]] = None):
        super().__init__(data_source=data_source)
        self.ds_len = len(data_source)
        if self.ds_len < 1:
            raise ValueError("dataset length must be at least 1")
        self.indices = [i for i in range(self.ds_len)]
        self.shuffle = shuffle
        self.reset_fn = reset_fn
        self.convert_fn = convert_fn
        self.idx = 0

    def __len__(self):
        return self.ds_len

    def __iter__(self):
        self.idx = 0
        if self.reset_fn:
            self.reset_fn(self.shuffle)
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.idx == self.ds_len:
            self.idx = 0
            if self.reset_fn:
                self.reset_fn(self.shuffle)
            if self.shuffle:
                random.shuffle(self.indices)
        elem = self.indices[self.idx]
        self.idx += 1
        if self.convert_fn:
            elem = self.convert_fn(elem)
        return elem


def decollate_batch(batch: Any) -> Any:
    """A function which takes a collated batch of data and turns it back into a list of instances.

    This method makes a number of implicit assumptions about the input data based on the fact that such data was
    previously run through the pytorch default_collate function.

    Args:
        batch: The collated batch of data.

    Returns:
        A list of data instances.
    """
    if isinstance(batch, str):
        return batch
    elif isinstance(batch, torch.Tensor):
        return [t.item() if t.ndim == 0 else t.numpy() for t in torch.unbind(batch)]
    elif isinstance(batch, dict):
        return [{k: v[idx] for idx, k in enumerate(batch.keys())} for v in
                zip(*[decollate_batch(b) for b in batch.values()])]
    elif isinstance(batch, (list, tuple)):
        transposed = [decollate_batch(elem) for elem in batch]
        if transposed and isinstance(transposed[0], str):
            pass  # Strings don't get extra batch dimension, so don't need special treatment
        else:
            transposed = [list(e) for e in zip(*transposed)]
        if hasattr(batch, '_fields'):  # Named tuple
            transposed = [type(batch)(*e) for e in transposed]
        return transposed
    raise TypeError("Didn't implement this yet")
