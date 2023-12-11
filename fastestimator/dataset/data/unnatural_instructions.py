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
import os
import zipfile
from pathlib import Path
from typing import Optional

import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None) -> str:
    """Download Unnatural Instruction tuning data and return its downloaded file path.

    The data contains 68,478 instruction-output pairs. More in https://github.com/orhonovich/unnatural-instructions.

    Args:
        root_dir: Download parent path. Defaults to None.

    Returns:
        str: Json file path.
    """
    # Set up path
    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'unnatural_instructions')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'unnatural_instructions')
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "core_data.zip")
    # download data
    if not os.path.exists(zip_path):
        print("Downloading data to {}".format(zip_path))
        wget.download("https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip",
                      zip_path,
                      bar=bar_custom)
    extracted_file = os.path.join(root_dir, "core_data.jsonl")
    # extract data
    if not os.path.exists(extracted_file):
        print("Extracting {}".format(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(root_dir)
    return extracted_file
