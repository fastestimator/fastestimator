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
from copy import deepcopy
from typing import Any, List, Mapping

import numpy as np
from torch.utils.data import Dataset

from fastestimator.dataset import BatchDataset
from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import pad_batch


@traceable()
class OpDataset(Dataset):
    """A wrapper for datasets which allows operators to be applied to them in a pipeline.

    This class should not be directly instantiated by the end user. The fe.Pipeline will automatically wrap datasets
    within an Op dataset as needed.

    Args:
        dataset: The base dataset to wrap.
        ops: A list of ops to be applied after the base `dataset` `__getitem__` is invoked.
        mode: What mode the system is currently running in ('train', 'eval', 'test', or 'infer').
    """
    def __init__(self, dataset: Dataset, ops: List[NumpyOp], mode: str) -> None:
        self.dataset = dataset
        if isinstance(self.dataset, BatchDataset):
            self.dataset.reset_index_maps()
        self.ops = ops
        self.mode = mode

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Fetch a data instance at a specified index, and apply transformations to it.

        Args:
            index: Which datapoint to retrieve.

        Returns:
            The data dictionary from the specified index, with transformations applied.
        """
        items = deepcopy(self.dataset[index])  # Deepcopy to prevent ops from overwriting values in datasets
        if isinstance(self.dataset, BatchDataset):
            unique_list = []
            for item in items:
                if id(item) not in unique_list:
                    forward_numpyop(self.ops, item, self.mode)
                    unique_list.append(id(item))
            if self.dataset.pad_value is not None:
                pad_batch(items, self.dataset.pad_value)
            items = {key: np.array([item[key] for item in items]) for key in items[0]}
        else:
            forward_numpyop(self.ops, items, self.mode)
        return items

    def __len__(self):
        return len(self.dataset)
