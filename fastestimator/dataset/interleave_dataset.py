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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union, overload

from torch.utils.data import Dataset

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.util.base_util import to_list, warn
from fastestimator.util.traceability_util import traceable


@traceable()
class InterleaveDataset(FEDataset):
    """A Dataset class that can allow for a step-wise interleaving of multiple datasets.

    This will be useful for training multi-task models when we vary dataset used on a per-step basis.

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

    When datasets are provided as a dictionary, users can use its key as `ds_id` in other Pipeline Operators to apply
    dataset-specific operations (such as batching or preprocessing). For example, if we need `ds1` to go through
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
    @overload
    def __init__(self, dataset: List[Dataset], pattern: Optional[List[int]]) -> None:
        ...

    @overload
    def __init__(self, dataset: Mapping[str, Dataset], pattern: Optional[List[str]]) -> None:
        ...

    def __init__(self,
                 datasets: Union[Mapping[str, Dataset], List[Dataset]],
                 pattern: Optional[Union[List[str], List[int]]] = None) -> None:
        self.datasets = datasets
        self.pattern = pattern
        self.tags = None
        self.batch_sizes = None
        self.fe_batch = None  # fe_batch member variable is used by FEDataLoader
        self.op_datasets = None
        self.all_fe_datasets = False
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
                assert all([p in tags for p in self.pattern]), "pattern name(s) missing in dataset dictionary, \
                    patterns are: {}, available names are: {}".format(self.pattern, tags)
                self.pattern = [self.tags.index(p) for p in self.pattern]
        elif isinstance(self.datasets, list):
            self.tags = [None for _ in self.datasets]
            if self.pattern:
                available_idx = [x for x in range(len(self.datasets))]
                assert all([p in available_idx for p in self.pattern]), "pattern index not matching with dataset, \
                    patterns are: {}, available indexes are: {}".format(self.pattern, available_idx)
        else:
            raise ValueError("datasets must be either a list or dictionary")
        # fill in default interleaving pattern
        if self.pattern is None:
            self.pattern = [x for x in range(len(self.datasets))]
        # batch size information will be altered by pipeline before training.
        self.batch_sizes = [1 for _ in self.datasets]
        assert len(self.datasets) > 1, "require more than one dataset as input of InterleaveDataset"
        assert len(set(self.pattern)) == len(self.datasets), "not all datasets are mentioned in the pattern"
        self.frequency = [self.pattern.count(idx) for idx in range(len(self.datasets))]
        self.all_fe_datasets = all([isinstance(dataset, FEDataset) for dataset in self.datasets])

    def set_batch_sizes(self, batch_sizes: List[int]) -> None:
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
        # first calculate the minimum number of cycles each dataset can afford according to the repeat pattern
        num_cycles = min(len(ds) // (f * bs) for ds, f, bs in zip(self.datasets, self.frequency, self.batch_sizes))
        assert num_cycles > 0, "some dataset does not have enough samples for a single repeat pattern, please consider \
            using `ExtendDataset` to increase its length"
        # returning the sum of number of batches of each dataset
        return sum(num_cycles * f for f in self.frequency)

    def __getitem__(self, batch_idx: int) -> List[Dict[str, Any]]:
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

    def split(self,
              *fractions: Union[float, int, Iterable[int]],
              seed: Optional[int] = None,
              stratify: Optional[str] = None) -> Union['InterleaveDataset', List['InterleaveDataset']]:
        """Split this dataset into multiple smaller datasets.

        This function enables several types of splitting:
        1. Splitting by fractions.
            ```python
            ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
            ds2 = ds.split(0.1)  # len(ds) == 900, len(ds2) == 100
            ds3, ds4 = ds.split(0.1, 0.2)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
            ```
        2. Splitting by counts.
            ```python
            ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
            ds2 = ds.split(100)  # len(ds) == 900, len(ds2) == 100
            ds3, ds4 = ds.split(90, 180)  # len(ds) == 630, len(ds3) == 90, len(ds4) == 180
            ```
        3. Splitting by indices.
            ```python
            ds = fe.dataset.FEDataset(...)  # len(ds) == 1000
            ds2 = ds.split([87,2,3,100,121,158])  # len(ds) == 994, len(ds2) == 6
            ds3 = ds.split(range(100))  # len(ds) == 894, len(ds3) == 100
            ```

        Args:
            *fractions: Floating point values will be interpreted as percentages, integers as an absolute number of
                datapoints, and an iterable of integers as the exact indices of the data that should be removed in order
                to create the new dataset.
            seed: The random seed to use when splitting the dataset. Useful if you want consistent splits across
                multiple experiments. This isn't necessary if you are splitting by data index.
            stratify: A class key within the dataset with which to stratify the split (to approximately maintain class
                balance ratios before and after a split). Incompatible with data index splitting.

        Returns:
            One or more new datasets which are created by removing elements from the current dataset. The number of
            datasets returned will be equal to the number of `fractions` provided. If only a single value is provided
            then the return will be a single dataset rather than a list of datasets.

        Raises:
            NotImplementedError: If the user created this dataset using one or more non-FEDataset inputs.
        """
        if not self.all_fe_datasets:
            raise NotImplementedError(
                "InterleaveDataset.split() is not supported when InterleaveDataset contains non-FEDataset objects")
        # Only pass the stratify argument to the dataset(s) which have the appropriate key
        new_datasets = [
            to_list(ds.split(*fractions, seed=seed, stratify=stratify if stratify in ds[0] else None))
            for ds in self.datasets
        ]
        num_splits = len(new_datasets[0])
        new_datasets = [[ds[i] for ds in new_datasets] for i in range(num_splits)]
        results = [InterleaveDataset(ds, pattern=self.pattern) for ds in new_datasets]
        if seed is not None:
            for ds in results:
                ds.fe_reset_ds(seed=seed)
        # Re-compute personal variables
        self.fe_reset_ds(seed=seed)
        FEDataset.fix_split_traceabilty(self, results, fractions, seed, stratify)
        # Unpack response if only a single split
        if len(results) == 1:
            results = results[0]
        return results

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'datasets': [ds.__getstate__() if hasattr(ds, '__getstate__') else {} for ds in self.datasets]}

    def summary(self) -> DatasetSummary:
        """Generate a summary representation of this dataset.
        Returns:
            A summary representation of this dataset.
        """
        if not self.all_fe_datasets:
            warn("InterleaveDataset summary will be incomplete since non-FEDatasets were used.")
            return DatasetSummary(num_instances=len(self), keys={})
        summaries = [ds.summary() for ds in self.datasets]
        keys = {k: v for summary in summaries for k, v in summary.keys.items()}
        return DatasetSummary(num_instances=len(self), keys=keys)
