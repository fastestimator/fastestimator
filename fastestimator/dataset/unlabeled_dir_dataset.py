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
from copy import deepcopy
from typing import Optional, Dict, Sequence, Iterable, Any, List

from fastestimator.dataset.dataset import FEDataset


class UnlabeledDirDataset(FEDataset):
    """ A dataset which reads files from a folder hierarchy

    Args:
        root_dir: The path to the directory containing data
        data_key: What key to assign to the data values in the data dictionary
        file_extension: If provided then only files ending with the file_extension will be included
        recursive_search: Whether to search within subdirectories for files
    """
    data: Dict[int, Dict[str, str]]

    def __init__(self,
                 root_dir: str,
                 data_key: str = "x",
                 file_extension: Optional[str] = None,
                 recursive_search: bool = True):
        data = []
        root_dir = os.path.normpath(root_dir)
        if not os.path.isdir(root_dir):
            raise AssertionError("Provided path is not a directory")
        try:
            for root, dirs, files in os.walk(root_dir):
                for file_name in files:
                    if file_name.startswith(".") or (file_extension is not None
                                                     and not file_name.endswith(file_extension)):
                        continue
                    data.append((os.path.join(root, file_name), os.path.basename(root)))
                if not recursive_search:
                    break
        except StopIteration:
            raise ValueError("Invalid directory structure for UnlabeledDirDataset at root: {}".format(root_dir))
        self.data = {i: {data_key: data[i][0]} for i in range(len(data))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return deepcopy(self.data[index])

    @classmethod
    def _skip_init(cls, data: Dict[int, Dict[str, Any]]) -> 'UnlabeledDirDataset':
        obj = cls.__new__(cls)
        obj.data = data
        return obj

    def _do_split(self, splits: Sequence[Iterable[int]]) -> List['UnlabeledDirDataset']:
        results = []
        for split in splits:
            data = {new_idx: self.data.pop(old_idx) for new_idx, old_idx in enumerate(split)}
            results.append(UnlabeledDirDataset._skip_init(data))
        # Re-key the remaining data to be contiguous from 0 to new max index
        self.data = {new_idx: v for new_idx, (old_idx, v) in enumerate(self.data.items())}
        return results
