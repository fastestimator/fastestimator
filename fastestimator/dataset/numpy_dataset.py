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
from typing import List, Iterable, Dict, Any, Sequence

import numpy as np

from fastestimator.dataset.fe_dataset import FEDataset


class NumpyDataset(FEDataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        size = None
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if size is not None:
                    assert val.shape[0] == size, "All data arrays must have the same number of elements"
                else:
                    size = val.shape[0]
        assert isinstance(size, int), \
            "Could not infer size of data. Please ensure you are passing numpy arrays in the data dictionary."
        self.data = {i: {k: v[i] for k, v in data.items()} for i in range(size)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def _skip_init(cls, data: Dict[int, Dict[str, Any]]) -> 'NumpyDataset':
        obj = cls.__new__(cls)
        obj.data = data
        return obj

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['NumpyDataset']:
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(NumpyDataset._skip_init(data))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        return results
