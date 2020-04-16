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
            raise ValueError("Invalid directory structure for DirDataset at root: {}".format(root_dir))
        super().__init__({i: {data_key: data[i][0]} for i in range(len(data))})
