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
from typing import Dict, List, Union

import numpy as np

from fastestimator.dataset.dataset import InMemoryDataset
from fastestimator.util.traceability_util import traceable


@traceable()
class NumpyDataset(InMemoryDataset):
    """A dataset constructed from a dictionary of Numpy data or list of data.

    Args:
        data: A dictionary of data like {"key1": <numpy array>, "key2": [list]}.
    Raises:
        AssertionError: If any of the Numpy arrays or lists have differing numbers of elements.
        ValueError: If any dictionary value is not instance of Numpy array or list.
    """
    def __init__(self, data: Dict[str, Union[np.ndarray, List]]) -> None:
        size = None
        for val in data.values():
            if isinstance(val, np.ndarray):
                current_size = val.shape[0]
            elif isinstance(val, list):
                current_size = len(val)
            else:
                raise ValueError("Please ensure you are passing numpy array or list in the data dictionary.")
            if size is not None:
                assert size == current_size, "All data arrays must have the same number of elements"
            else:
                size = current_size
        super().__init__({i: {k: v[i] for k, v in data.items()} for i in range(size)})
