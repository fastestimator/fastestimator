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
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from fastestimator.dataset.dataset import DatasetSummary, FEDataset
from fastestimator.util.util import to_list


class BatchDataset(FEDataset):
    """BatchDataset can create list of batch data from single dataset or multiple datasets.

    Args:
        datasets: FEDataset instance or list of FEDataset instances.
        num_sampling: number of times to do the sampling according the distribution.
        distribution: Number of samples or probability to draw from each dataset. it can be a list of integer or float.
                      Defaults to None, which will be [1, 1, ..., 1].

        Example:
        Given two datasets if the batch size is 8 and we want to draw 5 from first dataset and remaining 3 from the
        second: num_sampling = 1, distribution = [5, 3].

        Given two datasets if the batch size is 8 and we want to draw with 0.7 probability from first dataset and 0.3
        from the second: num_sampling = 8, distribution = [0.7, 0.3].
    """
    def __init__(self,
                 datasets: Union[FEDataset, Iterable[FEDataset]],
                 num_sampling: int,
                 distribution: Optional[Iterable[Union[int, float]]] = None):
        self.num_sampling = num_sampling
        self.datasets = to_list(datasets)
        self.index_maps = [list(range(len(dataset))) for dataset in self.datasets]
        if distribution:
            self.distribution = to_list(distribution)
        else:
            self.distribution = [1] * len(self.datasets)
        self.probability_mode = False
        self._check_input()

    def _check_input(self):
        assert len(self.datasets) == len(self.distribution), "the length of distribution and dataset must be consistent"
        if np.sum(self.distribution) == 1 and len(self.distribution) > 1:
            self.probability_mode = True
        for idx, dist in enumerate(self.distribution):
            if self.probability_mode:
                assert isinstance(dist, float) and dist > 0, "must provide positive float for probability distribution"
            else:
                assert isinstance(dist, int) and dist > 0, "must provide positive integer for sample distribution"
                # setting num_sampling as 1 for non-probability samoling
                self.distribution[idx] = self.num_sampling * dist

    def _do_split(self, splits: Sequence[Iterable[int]]):
        # Overwriting the split() method instead of _do_split
        raise AssertionError("This method should not have been invoked. Please file a bug report")

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['UnpairedDataset', List['UnpairedDataset']]:
        new_datasets = [to_list(ds.split(*fractions)) for ds in self.datasets]
        num_splits = len(new_datasets[0])
        new_datasets = [[ds[i] for ds in new_datasets] for i in range(num_splits)]
        results = [BatchDataset(ds, self.num_sampling, self.distribution) for ds in new_datasets]
        # Re-compute personal variables
        self.index_maps = [list(range(len(dataset))) for dataset in self.datasets]
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
        return DatasetSummary(num_instances=self.__len__(), keys=keys)

    def __len__(self) -> int:
        if self.probability_mode:
            length = np.max([len(dataset) // self.num_sampling for dataset in self.datasets])
        else:
            length = np.max([len(dataset) // dist for (dataset, dist) in zip(self.datasets, self.distribution)])
        return length

    def __getitem__(self, batch_idx: int) -> Dict[str, Any]:
        if self.probability_mode:
            index = list(np.random.choice(range(len(self.datasets)), size=self.num_sampling, p=self.distribution))
            distribution = [index.count(i) for i in range(len(self.datasets))]
        else:
            distribution = self.distribution
        items = []
        for dataset, num_sample, index_map in zip(self.datasets, distribution, self.index_maps):
            for idx in num_sample:
                items.append(dataset[index_map[(batch_idx * num_sample + idx) % len(dataset)]])
        return items
