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
from typing import Dict

import numpy as np

from fastestimator.dataset.dataset import InMemoryDataset


class NumpyDataset(InMemoryDataset):
    """A dataset constructed from a dictionary of Numpy data.

    Args:
        data: A dictionary of data like {"key": <numpy array>}.

    Raises:
        AssertionError: If any of the Numpy arrays have differing numbers of elements.
    """
    def __init__(self, data: Dict[str, np.ndarray]) -> None:
        size = None
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if size is not None:
                    assert val.shape[0] == size, "All data arrays must have the same number of elements"
                else:
                    size = val.shape[0]
        assert isinstance(size, int), \
            "Could not infer size of data. Please ensure you are passing numpy arrays in the data dictionary."
        super().__init__({i: {k: v[i] for k, v in data.items()} for i in range(size)})
