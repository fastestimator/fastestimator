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
from typing import Dict, Optional

from fastestimator.dataset.dataset import InMemoryDataset
from fastestimator.util.traceability_util import traceable
from fastestimator.util.base_util import list_files


@traceable()
class DirDataset(InMemoryDataset):
    """A dataset which reads files from a folder hierarchy like root/data.file.

    Args:
        root_dir: The path to the directory containing data.
        data_key: What key to assign to the data values in the data dictionary.
        file_extension: If provided then only files ending with the file_extension will be included.
        recursive_search: Whether to search within subdirectories for files.
    """
    data: Dict[int, Dict[str, str]]

    def __init__(self,
                 root_dir: str,
                 data_key: str = "x",
                 file_extension: Optional[str] = None,
                 recursive_search: bool = True) -> None:
        root_dir = os.path.normpath(root_dir)
        self.root_dir = root_dir
        data = list_files(root_dir=root_dir, file_extension=file_extension, recursive_search=recursive_search)
        # Sort the data so that deterministic split will work properly
        data.sort()
        super().__init__({i: {data_key: data[i]} for i in range(len(data))})
