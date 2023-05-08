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

from fastestimator.dataset.dataset import FEDataset


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
        # Potential checks
        # -- check if dataset is a type of pytorch dataset
        if len(datasets) < 2:
            raise AssertionError("Please provide atleast 2 datasets.")
        self.datasets = datasets
        # match the keys of 0th index item of all the datasets
        keys = datasets[0][0].keys()
        for ds in datasets[1:]:
            if ds[0].keys() != keys:
                raise KeyError("All datasets should have same keys.")

    def __len__(self):
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
