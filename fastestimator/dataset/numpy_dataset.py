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

import numpy as np
from torch.utils.data import Dataset


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
