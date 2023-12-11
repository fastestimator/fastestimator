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
from typing import Optional, Tuple

import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """Download Wikitext-103 data and return its downloaded file path.

    The training data contains 28475 wiki articles, 103 million tokens. The evaluation contains 60 wiki articles and
    240k tokens.

    Args:
        root_dir: Download parent path. Defaults to None.

    Returns:
        Tuple[str, str, str]: the file path for train, eval and test split.
    """
    # Set up path
    home = str(Path.home())
    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'wiki_text_103')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'wiki_text_103')
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "wikitext-103-raw-v1.zip")
    # download data
    if not os.path.exists(zip_path):
        print("Downloading data to {}".format(zip_path))
        wget.download("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
                      zip_path,
                      bar=bar_custom)
    extracted_folder = os.path.join(root_dir, "wikitext-103-raw")
    # extract data
    if not os.path.exists(extracted_folder):
        print("Extracting {}".format(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(root_dir)
    train_file = os.path.join(extracted_folder, "wiki.train.raw")
    eval_file = os.path.join(extracted_folder, "wiki.valid.raw")
    test_file = os.path.join(extracted_folder, "wiki.test.raw")
    return train_file, eval_file, test_file
