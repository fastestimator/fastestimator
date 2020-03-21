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
from fastestimator.util import to_list


class BatchDataset(FEDataset):
    """BatchDataset can create list of batch data from single dataset or multiple datasets.

    Args:
        datasets: FEDataset instance or list of FEDataset instances.
        num_samples: Number of samples to draw from dataset(s).
        probability: Probability to draw from each dataset. Defaults to None.

        Example:
        Given two datasets if the batch size is 8 and we want to randomly draw 5 from first dataset and remaining 3 from
         the second: num_samples = [5, 3].

        Given two datasets if the batch size is 8 and we want to draw with 0.7 probability from first dataset and 0.3
        from the second: num_samples = 8, probability = [0.7, 0.3].
    """
    def __init__(self,
                 datasets: Union[FEDataset, Iterable[FEDataset]],
                 num_samples: Union[int, Iterable[int]],
                 probability: Optional[Iterable[float]] = None):
        self.datasets = to_list(datasets)
        self.num_samples = to_list(num_samples)
        self.probability = to_list(probability)
        self.same_feature = False
        self._check_input()
        self.index_maps = [list(range(max((len(dataset) for dataset in self.datasets)))) for _ in self.datasets]

    def _check_input(self):
        for num_sample in self.num_samples:
            assert isinstance(num_sample, int) and num_sample > 0, "only accept positive integer type as num_sample"
        # check dataset keys
        dataset_keys = [set(dataset[0].keys()) for dataset in self.datasets]
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
        if not self.same_feature:
            assert len(set(self.num_samples)) == 1, "the number of samples must be the same for disjoint features"

    def _do_split(self, splits: Sequence[Iterable[int]]):
        # Overwriting the split() method instead of _do_split
        raise AssertionError("This method should not have been invoked. Please file a bug report")

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['UnpairedDataset', List['UnpairedDataset']]:
        new_datasets = [to_list(ds.split(*fractions)) for ds in self.datasets]
        num_splits = len(new_datasets[0])
        new_datasets = [[ds[i] for ds in new_datasets] for i in range(num_splits)]
        results = [BatchDataset(ds, self.num_samples, self.probability) for ds in new_datasets]
        # Re-compute personal variables
        self.index_maps = [list(range(max((len(dataset) for dataset in self.datasets)))) for _ in self.datasets]
        # Unpack response if only a single split
        if len(results) == 1:
            results = results[0]
        return results

    def shuffle(self):
        """
        This method is invoked every epoch by OpDataset which allows each epoch to have different random pairings of the
         basis datasets.
        """
        for mapping in self.index_maps:
            random.shuffle(mapping)

    def summary(self) -> DatasetSummary:
        summaries = [ds.summary() for ds in self.datasets]
        keys = {k: v for summary in summaries for k, v in summary.keys.items()}
        return DatasetSummary(num_instances=len(self), keys=keys)

    def __len__(self) -> int:
        if self.same_feature:
            length = math.ceil(sum([len(dataset) for dataset in self.datasets]) / sum(self.num_samples))
        else:
            num_sample = self.num_samples[0]
            length = max([math.ceil(len(ds) / num_sample) for ds in self.datasets])
        return length

    def __getitem__(self, batch_idx: int) -> List[Dict[str, Any]]:
        items = []
        if self.same_feature:
            if self.probability:
                index = list(np.random.choice(range(len(self.datasets)), size=self.num_samples, p=self.probability))
                num_samples = [index.count(i) for i in range(len(self.datasets))]
            else:
                num_samples = self.num_samples
            for dataset, num_sample, index_map in zip(self.datasets, num_samples, self.index_maps):
                for idx in range(num_sample):
                    items.append(dataset[index_map[(batch_idx * num_sample + idx)] % len(dataset)])
        else:
            num_sample = self.num_samples[0]
            for idx in range(num_sample):
                paired_items = [
                    dataset[index_map[(batch_idx * num_sample + idx)] % len(dataset)] for dataset,
                    index_map in zip(self.datasets, self.index_maps)
                ]
                items.append({k: v for d in paired_items for k, v in d.items()})
        return items
