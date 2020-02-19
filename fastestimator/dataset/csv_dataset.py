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
import os
from typing import Dict, Iterable, List, Any, Sequence

import pandas as pd

from fastestimator.dataset.fe_dataset import FEDataset


class CSVDataset(FEDataset):
    """ CSVDataset reads entries from a CSV file, where the first row is the header. The root directory of the csv file
         may be accessed using dataset.parent_path. This may be useful if the csv contains relative path information
         that you want to feed into, say, an ImageReader Op
    Args:
        csv_path: The (absolute) path to the CSV file
        delimiter: What delimiter is used by the file
        kwargs: Other arguments to be passed through to pandas csv reader function
            (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
    """
    def __init__(self, csv_path: str, delimiter: str = ",", **kwargs):
        df = pd.read_csv(csv_path, delimiter=delimiter, **kwargs)
        self.data = df.to_dict(orient='index')
        self.parent_path = os.path.dirname(csv_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]

    @classmethod
    def _skip_init(cls, data: Dict[int, Dict[str, Any]], parent_path: str) -> 'CSVDataset':
        obj = cls.__new__(cls)
        obj.data = data
        obj.parent_path = parent_path
        return obj

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['CSVDataset']:
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(CSVDataset._skip_init(data, self.parent_path))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        return results


class CSVDatasets:
    """ A class which instantiates multiple CSVDataset from a folder containing one or more .csv files
    Args:
        root_dir: The path to the directory containing CSV files
        delimiter: What delimiter is used by the file
        kwargs: Other arguments to be passed through to pandas csv reader function
            (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
    """
    datasets: Dict[str, CSVDataset]

    def __init__(self, root_dir: str, delimiter: str = ",", **kwargs):
        root_dir = os.path.normpath(root_dir)
        self.datasets = {}
        try:
            _, _, files = next(os.walk(root_dir))
            for file in files:
                if file.endswith(".csv"):
                    self.datasets[file[0:-4]] = CSVDataset(os.path.join(root_dir, file), delimiter=delimiter, **kwargs)
        except StopIteration:
            raise ValueError("Invalid directory structure for CSVDatasets at root: {}".format(root_dir))

    def __getitem__(self, mode: str) -> CSVDataset:
        return self.datasets[mode]
