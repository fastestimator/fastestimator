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

from typing import List

from torch.utils.data import ConcatDataset, Dataset

from fastestimator.dataset.interleave_dataset import InterleaveDataset
from fastestimator.util.traceability_util import traceable


@traceable()
class CombinedDataset(ConcatDataset):
    """Combines a list of PyTorch Datasets.

    Args:
        datasets: Pytorch (or FE) Datasets to be combined.

    Raises:
        AssertionError: raise exception when the input list has less than 2 datasets.
        KeyError: raise exception when the datasets does not have same keys.
    """
    def __init__(self, datasets: List[Dataset]) -> None:
        super().__init__(datasets)
        keys = None

        for ds in datasets:
            if isinstance(ds, InterleaveDataset):
                raise AssertionError("CombinedDataset does not support InterleaveDataset")
            if isinstance(ds, Dataset) and isinstance(ds[0], dict):
                if keys is None:
                    keys = ds[0].keys()
                elif ds[0].keys() != keys:
                    raise KeyError("All datasets should have the same keys.")
            else:
                raise AssertionError("Each dataset should be a type of PyTorch Dataset and should return a dictionary.")
