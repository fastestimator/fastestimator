# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
import random
from typing import Any, Dict, List, Mapping, Optional, Union

from torch.utils.data import Dataset

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.util.traceability_util import traceable


@traceable()
class InterleaveDataset(FEDataset):
    """A Dataset class that can allow for a step-wise interleaving of multiple datasets. This will be useful for
    training multi-task models when we vary dataset used on a per-step basis.

    For example, given dataset `ds1`, and `ds2`, if we want to vary dataset between each step, the following 3 will
    produce the same behavior: step0 getting samples from ds1, step1 getting samples from ds2, and repeat.
    ```python
    dataset = InterleaveDataset(datasets=[ds1, ds2])
    dataset = InterleaveDataset(datasets=[ds1, ds2], pattern=[0, 1])
    dataset = InterleaveDataset(datasets={"a": ds1, "b": ds2}, pattern=["a", "b"])
    ```

    To achieve a more complicated interleaving pattern, for example, 2 steps of `ds1` followed by 3 step of `ds2`:
    ```python
    dataset = InterleaveDataset(datasets=[ds1, ds2], pattern=[0, 0, 1, 1, 1])
    dataset = InterleaveDataset(datasets={"a": ds1, "b": ds2}, pattern=["a", "a", "b", "b", "b"])
    ```

    When datasets are provided as dictionary, users can use its key as `ds_id` in other Pipeline Operators to apply
    dataset-spefic operations (such as batching or preprocessing). For example, if we need `ds1` to go through
    `Minmax` then form a batch of 32, and `ds2` to use `Zscore` then form a batch of 64:
    ```python
    dataset = InterleaveDataset(datasets={"a": ds1, "b": ds2}, pattern=["a", "b"])
    pipeline = fe.Pipeline(train_data = dataset,
                            ops=[Minmax(..., ds_id="a"),
                                Zscore(..., ds_id="b"),
                                Batch(batch_size=32, ds_id="a"),
                                Batch(batch_size=64, ds_id="b")])
    ```

    Args:
        datasets: List of datasets or a dictionary with key being the dataset name, value being the dataset.
        pattern: The step-wise interleaving patterns. When datasets provided is a list, it requires list of integer
            when `datasets` is list, requires a list of names when `datasets` is dictionary.
    """
    def __init__(self,
                 datasets: Union[Mapping[str, Dataset], List[Dataset]],
                 pattern: Optional[Union[List[str], List[int]]] = None) -> None:
        self.datasets = datasets
        self.pattern = pattern
        self.tags = None
        self.batch_sizes = None
        self.fe_batch = None  # fe_batch member variable is used by FEDataLoader
        self.batch_sizes_overwritten = False
        self.warned = False
        self.op_datasets = None
        self.index_maps = []
        self._standardize_args()
        self.child_reset_fns = [ds.fe_reset_ds for ds in self.datasets if hasattr(ds, 'fe_reset_ds')]
        self.fe_reset_ds(seed=0)  # using a seed here to ensure the same initial condition for possible seeded split.

    def _standardize_args(self) -> None:
        """Convert the dictionary based datasets into list-based dataset, and fill in some attributes.
        """
        if self.pattern is not None:
            assert isinstance(self.pattern, list), "pattern must be a list"
            assert len(self.pattern) > 1, "length of pattern must be greater than 1"
        # dataset, tag, pattern standardization
        if isinstance(self.datasets, dict):
            datasets, tags = [], []
            for k, v in self.datasets.items():
                assert isinstance(k, str), "each dataset key must be a string, found: {}".format(k)
                tags.append(k)
                datasets.append(v)
            self.datasets = datasets
            self.tags = tags
            # standardize its pattern too if it exists
            if self.pattern:
                assert all([p in tags for p in self.pattern]), "pattern name(s) missing in dataset dictionary, patterns are: {}, available names are: {}".format(self.pattern, tags)
                self.pattern = [self.tags.index(p) for p in self.pattern]
        elif isinstance(self.datasets, list):
            self.tags = [None for _ in self.datasets]
            if self.pattern:
                available_idx = [x for x in range(len(self.datasets))]
                assert all([p in available_idx for p in self.pattern]), "pattern index not matching with dataset, patterns are: {}, available indexes are: {}".format(self.pattern, available_idx)
        else:
            raise ValueError("datasets must be either a list or dictionary")
        # fill in default interleaving pattern
        if self.pattern is None:
            self.pattern = [x for x in range(len(self.datasets))]
        # batch size information will be altered by pipeline before training.
        self.batch_sizes = [1 for _ in self.datasets]
        assert len(self.datasets) > 1, "require more than one dataset as input of InterleaveDataset"
        assert len(set(self.pattern)) == len(self.datasets), "not all dataset is available in the pattern"
        self.frequency = [self.pattern.count(idx) for idx in range(len(self.datasets))]

    def set_batch_sizes(self, batch_sizes: List[int]) -> None:
        self.batch_sizes_overwritten = True
        self.batch_sizes = batch_sizes
        self.fe_batch = batch_sizes

    def fe_reset_ds(self, shuffle: bool = True, *, seed: Optional[int] = None) -> None:
        """Rearrange the index maps of this InterleaveDataset.

        Args:
            shuffle: Whether to shuffle the dataset. If False the method will do nothing so long as index maps already
                exist.
            seed: A random seed to control the shuffling. This is provided for compatibility with the dataset.split
                method random seed. It's not necessary from a training functionality perspective since shuffling is
                performed every epoch, but if user wants to visualize a dataset element after the split this will help.

        This method is invoked by the FEDataLoader which allows each epoch to have different random pairings of the
        basis datasets.
        """
        # Reset any children who need resetting
        for fn in self.child_reset_fns:
            fn(shuffle=shuffle, seed=seed)
        # Don't bother re-initializing if shuffle is False
        if shuffle is False and self.index_maps:
            return
        self.index_maps = []
        for idx, ds in enumerate(self.datasets):
            index_map = list(range(len(ds)))
            if seed is not None:
                # adding idx to the seed because we need to make sure different datasets have different index orders,
                # in the meantime, their random behavior should still be conditioned on seed.
                random.Random(seed + idx).shuffle(index_map)
            else:
                random.shuffle(index_map)
            if hasattr(ds, "fe_batch_indices"):
                self.index_maps.append([ds.fe_batch_indices(index) for index in index_map])
            else:
                self.index_maps.append(index_map)

    def __len__(self) -> int:
        """Compute the length of this dataset.

        Returns:
            How many batches of data can this dataset serve per epoch. It is sum of all dataset's number of batch.
        """
        if not self.batch_sizes_overwritten and not self.warned:
            print("Fastestimator-Warn: The length of the InterleaveDataset depends on the batch size.")
            self.warned = True
        # first calcualte the minimum number of cycles each dataset can afford according to the repeat pattern
        num_cycles = min(len(ds) // (f * bs) for ds, f, bs in zip(self.datasets, self.frequency, self.batch_sizes))
        assert num_cycles > 0, "some dataset does not have enough samples for a single repeat pattern, please consider using `ExtendDataset` to increase its length"
        # returning the sum of number of batches of each dataset
        return sum(num_cycles * f for f in self.frequency)

    def __getitem__(self, batch_idx: int) -> Dict[str, Any]:
        """Extract items from the underlying datasets based on the given `batch_idx`.

        Args:
            batch_idx: Which the index of current batch to pull data from (or which batch_idx to query).

        Raises:
            StopIteration: when batch_idx is beyond the dataset length

        Returns:
            A list of data instance dictionaries corresponding to the current `batch_idx`. When plugged in Pipeline,
            this dataset will return the results of operators.
        """
        if batch_idx >= len(self):
            raise StopIteration
        cumulative_cycles, cycle_pos = batch_idx // len(self.pattern), batch_idx % len(self.pattern)
        ds_idx_now = self.pattern[cycle_pos]
        batch_size_now, index_map_now = self.batch_sizes[ds_idx_now], self.index_maps[ds_idx_now]
        dataset_now = self.datasets[ds_idx_now] if self.op_datasets is None else self.op_datasets[ds_idx_now]
        sample_start_idx = cumulative_cycles * self.frequency[ds_idx_now] * batch_size_now
        if cycle_pos > 0:
            # if in middle of cycle, need to account for the previously used samples
            sample_start_idx = sample_start_idx + sum(
                [self.batch_sizes[ds_idx_now] for ds_idx in self.pattern[:cycle_pos] if ds_idx == ds_idx_now])
        batch = []
        for idx in range(sample_start_idx, sample_start_idx + batch_size_now):
            item = dataset_now[index_map_now[idx]]
            if isinstance(item, list):
                batch.extend(item)
            else:
                batch.append(item)
        return batch
