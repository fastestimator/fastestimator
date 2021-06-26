# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import tarfile
from pathlib import Path
from typing import Optional, Tuple

import wget
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None, translate_option: str = "az_to_en"):
    """Load and return the neural machine translation dataset from TED talks.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        translate_option: options for translation languages, available options are: "az_to_en", "az_tr_to_en",
            "be_ru_to_en", "be_to_en", "es_to_pt", "fr_to_pt", "gl_pt_to_en", "gl_to_en", "he_to_pt", "it_to_pt",
            "pt_to_en", "ru_to_en", "ru_to_pt", "tr_to_en".

    Returns:
        (train_data, eval_data)
    """
    # Set up path
    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'tednmt')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'tednmt')
    os.makedirs(root_dir, exist_ok=True)
    compressed_path = os.path.join(root_dir, 'qi18naacl-dataset.tar.gz')
    extracted_path = os.path.join(root_dir, 'datasets')
    if not os.path.exists(extracted_path):
        # Download
        if not os.path.exists(compressed_path):
            print("Downloading data to {}".format(compressed_path))
            wget.download('http://www.phontron.com/data/qi18naacl-dataset.tar.gz', compressed_path, bar=bar_custom)
        # Extract
        print("\nExtracting files ...")
        with tarfile.open(compressed_path) as f:
            f.extractall(root_dir)
    # process data