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


from typing import Any, Dict, Union

from torch.utils.data import Dataset

from fastestimator.dataset.dataset import FEDataset
from fastestimator.util.traceability_util import traceable


@traceable()
class CombinedDataset(FEDataset):
    def __init__(self, datasets: list) -> None:
        """
        Combines a list of PyTorch datasets

        Args:
            datasets (list): List of PyTorch Datasets

        Raises:
            AssertionError: raise exception when the input list has less than 2 datasets.
            KeyError: raise exception when the datasets does not have same keys.
        """
        if not isinstance(datasets, list) or len(datasets) < 2:
            raise AssertionError("Please provide a list of atleast 2 datasets.")
        self.datasets = datasets
        keys = None

        for ds in datasets:
            if isinstance(ds, Dataset) and isinstance(ds[0], dict):
                if keys is None:
                    keys = ds[0].keys()
                elif ds[0].keys() != keys:
                    raise KeyError("All datasets should have same keys.")
            else:
                raise AssertionError(
                    "Each dataset should be a type of PyTorch Dataset and should return a dictionary."
                )

    def __len__(self) -> int:
        """
        Return combined length of datasets

        Returns:
            int: sum of the lengths of all datasets.
        """
        return sum([len(ds) for ds in self.datasets])

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], None]:
        """
        Return data based on the input id.

        Args:
            idx (int): index of the data item to be returned

        Returns:
            dict: dict at the index `idx`
        """
        start = 0
        end = 0
        if idx >= len(self):
            raise AssertionError(
                "Index is out of range of the length of the dataset. Please provide a valid index."
            )
        for ds in self.datasets:
            end += len(ds)
            if idx >= start and idx < end:
                return ds[idx - start]
            start += end
