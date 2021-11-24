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
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.dataset.extend_dataset import ExtendDataset
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_list


@traceable()
class BatchDataset(FEDataset):
    """BatchDataset extracts a list (batch) of data from a single dataset or multiple datasets.

    This dataset helps to enable several use-cases:
    1. Creating an unpaired dataset from two or more completely disjoint (no common keys) datasets.
        ```python
        ds1 = fe.dataset.DirDataset(...)  # {"a": <32x32>}
        ds2 = fe.dataset.DirDataset(...)  # {"b": <28x28>}
        unpaired_ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[4, 4])
        # {"a": <4x32x32>, "b": <4x28x28>}
        ```
    2. Deterministic class balanced sampling from two or more similar (all keys in common) datasets.
        ```python
        class1_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
        class2_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
        ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[3, 5])
        # {"x": <8x32x32>, "y": <8>}  (3 of the samples are from class1_ds, 5 of the samples from class2_ds)
        ```
    3. Probabilistic class balanced sampling from two or more similar (all keys in common) datasets.
        ```python
        class1_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
        class2_ds = fe.dataset.DirDataset(...)  # {"x": <32x32>, "y": <>}
        ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=8, probability=[0.7, 0.3])
        # {"x": <8x32x32>, "y": <8>}  (~70% of the samples are from class1_ds, ~30% of the samples from class2_ds)
        ```

    Args:
        datasets: The dataset(s) to use for batch sampling. While these should be FEDatasets, pytorch datasets will
            technically also work. If you use them, however, you will lose the .split() and .summary() methods.
        num_samples: Number of samples to draw from the `datasets`. May be a single int if used in conjunction with
            `probability`, otherwise a list of ints of len(`datasets`) is required.
        probability: Probability to draw from each dataset. Only allowed if `num_samples` is an integer.
    """
    def __init__(self,
                 datasets: Union[FEDataset, Iterable[FEDataset]],
                 num_samples: Union[int, Iterable[int]],
                 probability: Optional[Iterable[float]] = None) -> None:
        self.datasets = to_list(datasets)
        self.num_samples = to_list(num_samples)
        self.probability = to_list(probability)
        self.same_feature = False
        self.all_fe_datasets = False
        self._check_input()
        self.index_maps = []
        self.child_reset_fns = [dataset.fe_reset_ds for dataset in self.datasets if hasattr(dataset, 'fe_reset_ds')]
        self.fe_reset_ds(seed=0)

    def _check_input(self) -> None:
        """Verify that the given input values are valid.

        Raises:
            AssertionError: If any of the parameters are found to by unacceptable for a variety of reasons.
        """
        assert len(self.datasets) > 1, "must provide multiple datasets as input"
        for num_sample in self.num_samples:
            assert isinstance(num_sample, int) and num_sample > 0, "only accept positive integer type as num_sample"
        # check dataset keys
        dataset_keys = []
        num_examples = self.num_samples * len(self.datasets) if len(
            self.num_samples) == 1 else [x for x in self.num_samples]
        for idx, dataset in enumerate(self.datasets):
            sample_data = dataset[0]
            if isinstance(sample_data, list):
                keys = [set(sample_data_element.keys()) for sample_data_element in sample_data]
                keys = set.union(*keys)
                num_examples[idx] *= len(sample_data)
            else:
                keys = set(sample_data.keys())
            dataset_keys.append(keys)
        for key in dataset_keys:
            assert key, "found no key in datasets"
        is_same_key = all([dataset_keys[0] == key for key in dataset_keys])
        is_disjoint_key = sum([len(key) for key in dataset_keys]) == len(set.union(*dataset_keys))
        if len(self.datasets) > 1:
            assert is_same_key != is_disjoint_key, "dataset keys must be all same or all disjoint"
        self.same_feature = is_same_key
        if self.probability:
            assert self.same_feature, "keys must be exactly same among datasets when using probability distribution"
            assert len(self.datasets) == len(self.probability), "the length of dataset must match probability"
            assert len(self.num_samples) == 1, "num_sample must be scalar for probability mode"
            assert len(self.datasets) > 1, "number of datasets must be more than one to use probability mode"
            assert sum(self.probability) == 1, "sum of probability must be 1"
            for p in self.probability:
                assert isinstance(p, float) and p > 0, "must provide positive float for probability distribution"
        else:
            assert len(self.datasets) == len(self.num_samples), "the number of dataset must match num_samples"
        # set up batch size
        if self.same_feature:
            if self.probability:
                self.fe_batch = round(sum([n * p for n, p in zip(num_examples, self.probability)]))
            else:
                self.fe_batch = sum(num_examples)
        else:
            assert len(set(num_examples)) == 1, "the number of output samples must be the same for disjoint features"
            self.fe_batch = num_examples[0]
        self.all_fe_datasets = all([isinstance(dataset, FEDataset) for dataset in self.datasets])
        # Check ExtendDataset
        for idx, dataset in enumerate(self.datasets):
            assert not isinstance(dataset, ExtendDataset), "Input Dataset cannot be an ExtendDataset object"

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['BatchDataset']:
        """This class overwrites the .split() method instead of _do_split().

        Args:
            splits: Which indices to remove from the current dataset in order to create new dataset(s). One dataset will
                be generated for every element of the `splits` sequence.

        Raises:
            AssertionError: This method should never by invoked.
        """
        raise AssertionError("This method should not have been invoked. Please file a bug report")

    def split(self,
              *fractions: Union[float, int, Iterable[int]],
              seed: Optional[int] = None,
              stratify: Optional[str] = None) -> Union['BatchDataset', List['BatchDataset']]:
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
                "BatchDataset.split() is not supported when BatchDataset contains non-FEDataset objects")
        # Only pass the stratify argument to the dataset(s) which have the appropriate key
        new_datasets = [
            to_list(ds.split(*fractions, seed=seed, stratify=stratify if stratify in ds[0] else None))
            for ds in self.datasets
        ]
        num_splits = len(new_datasets[0])
        new_datasets = [[ds[i] for ds in new_datasets] for i in range(num_splits)]
        results = [BatchDataset(ds, self.num_samples, self.probability) for ds in new_datasets]
        if seed is not None:
            [ds.fe_reset_ds(seed=seed) for ds in results]
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
            print("FastEstimator-Warn: BatchDataset summary will be incomplete since non-FEDatasets were used.")
            return DatasetSummary(num_instances=len(self), keys={})
        summaries = [ds.summary() for ds in self.datasets]
        keys = {k: v for summary in summaries for k, v in summary.keys.items()}
        return DatasetSummary(num_instances=len(self), keys=keys)

    def __len__(self) -> int:
        """Compute the length of this dataset.
        Returns:
            How many batches of data can this dataset serve per epoch.
        """
        if len(self.num_samples) > 1:
            length = max([math.ceil(len(ds) / num_sample) for ds, num_sample in zip(self.datasets, self.num_samples)])
        else:
            num_sample = self.num_samples[0]
            length = max([math.ceil(len(ds) / num_sample / p) for ds, p in zip(self.datasets, self.probability)])
        return length

    def __getitem__(self, indices: Union[int, List[List[int]]]) -> List[Dict[str, Any]]:
        """Extract items from the underlying datasets based on the given `batch_idx`.

        Args:
            indices: Which indices to pull data from (or which batch_idx to query).

        Returns:
            A list of data instance dictionaries corresponding to the current `batch_idx`.
        """
        if isinstance(indices, int):
            indices = self.fe_batch_indices(indices)
        if self.same_feature:
            batch = []
            for dataset, idx_list in zip(self.datasets, indices):
                for idx in idx_list:
                    item = dataset[idx]
                    if isinstance(item, list):
                        batch.extend(item)
                    else:
                        batch.append(item)
        else:
            unpaired_items = []
            for dataset, idx_list in zip(self.datasets, indices):
                single_ds_items = []
                for idx in idx_list:
                    item = dataset[idx]
                    if isinstance(item, list):
                        single_ds_items.extend(item)
                    else:
                        single_ds_items.append(item)
                unpaired_items.append(single_ds_items)
            batch = [{k: v for d in d_pair for k, v in d.items()} for d_pair in zip(*unpaired_items)]
        random.shuffle(batch)
        return batch

    def fe_batch_indices(self, batch_idx: int) -> List[List[int]]:
        """Compute which internal dataset indices to use for a given batch.

        This method is separate from the __getitem__ call so that multi-processing can work correctly when data is
        filtered or extended.

        Args:
            batch_idx: Which batch is it.

        Returns:
            A list of data instance dictionaries corresponding to the current `batch_idx`.
        """
        if self.probability:
            index = list(np.random.choice(range(self.n_datasets), size=self.num_samples, p=self.probability))
            num_samples = [index.count(i) for i in range(self.n_datasets)]
        else:
            num_samples = self.num_samples
        indices = [[index_map[batch_idx * num_sample + idx] for idx in range(num_sample)] for num_sample, index_map
                   in zip(num_samples, self.index_maps)]
        return indices

    def fe_reset_ds(self, shuffle: bool = True, *, seed: Optional[int] = None) -> None:
        """Rearrange the index maps of this BatchDataset.

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
        num_samples = self.num_samples
        if self.probability:
            num_samples = num_samples * len(self.datasets)
        self.index_maps = []
        for dataset, num_sample in zip(self.datasets, num_samples):
            index_map = [list(range(len(dataset))) for _ in range(math.ceil(len(self) * num_sample / len(dataset)))]
            for mapping in index_map:
                if seed is not None:
                    random.Random(seed).shuffle(mapping)
                else:
                    random.shuffle(mapping)
            if hasattr(dataset, "fe_batch_indices"):
                self.index_maps.append([dataset.fe_batch_indices(item) for sublist in index_map for item in sublist])
            else:
                self.index_maps.append([item for sublist in index_map for item in sublist])
