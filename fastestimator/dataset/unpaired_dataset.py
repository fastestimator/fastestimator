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
from typing import Dict, Any, Sequence, Iterable, List, Union

from fastestimator.dataset.dataset import FEDataset, DatasetSummary
from fastestimator.util.util import to_list


class UnpairedDataset(FEDataset):
    def __init__(self, *datasets: FEDataset):
        assert len(datasets) > 1, "UnpairedDataset requires at least 2 datasets"
        self.datasets = datasets
        self.len = max((len(dataset) for dataset in datasets))
        self._verify_no_key_overlap(0)
        self.index_maps = [list(range(self.len)) for _ in range(len(datasets))]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        items = [dataset[self.index_maps[ea][index] % len(dataset)] for ea, dataset in enumerate(self.datasets)]
        return {k: v for d in items for k, v in d.items()}

    def _verify_no_key_overlap(self, index: int):
        items = [dataset[index] for dataset in self.datasets]
        results = {}
        for item in items:
            for key, val in item.items():
                assert key not in results, \
                    "UnpairedDatasets must have disjoint keys, but multiple datasets returned the key: {}".format(key)
                results[key] = val

    def shuffle(self):
        """
        This method is invoked every epoch by OpDataset which allows each epoch to have different random pairings of the
         basis datasets.
        """
        for mapping in self.index_maps:
            random.shuffle(mapping)

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['UnpairedDataset']:
        # Overwriting the split() method instead of _do_split
        raise AssertionError("This method should not have been invoked. Please file a bug report")

    def split(self, *fractions: Union[float, int, Iterable[int]]) -> Union['UnpairedDataset', List['UnpairedDataset']]:
        new_datasets = [to_list(ds.split(*fractions)) for ds in self.datasets]
        num_splits = len(new_datasets[0])
        new_datasets = [[ds[i] for ds in new_datasets] for i in range(num_splits)]
        results = [UnpairedDataset(*ds) for ds in new_datasets]
        # Re-compute personal variables
        self.len = max((len(dataset) for dataset in self.datasets))
        self.index_maps = [list(range(self.len)) for _ in range(len(self.datasets))]
        # Unpack response if only a single split
        if len(results) == 1:
            return results[0]
        else:
            return results

    def summary(self) -> DatasetSummary:
        summaries = [ds.summary() for ds in self.datasets]
        keys = {k: v for summary in summaries for k, v in summary.keys.items()}
        return DatasetSummary(num_instances=self.len, keys=keys)
