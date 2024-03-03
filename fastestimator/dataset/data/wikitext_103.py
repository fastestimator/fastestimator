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
from pathlib import Path
from typing import Optional, Tuple

import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def download_file(file_path: str, output_dir: str):
    filename = os.path.basename(file_path)
    output_location = os.path.join(output_dir, filename)
    if not os.path.exists(output_location):
        print("Downloading data to {}".format(filename))
        wget.download(file_path, output_location, bar=bar_custom)
    return output_location


def load_data(root_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """Download Wikitext-103 data and return its downloaded file path.

    The training data contains 28475 wiki articles, 103 million tokens. The evaluation contains 60 wiki articles and
    240k tokens. Since the original wikitext dataset url is no longer available, we are using dataset provided by
    huggingface datasets. The training dataset is provided as to parquet files and test and validation datasets are
    provided as single parquet file each. For simplicity we are providing only the first half of the training dataset with 900k rows.

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

    test_file = download_file(
        'https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet',
        root_dir)
    eval_file = download_file(
        'https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet',
        root_dir)
    train_file = download_file(
        "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
        root_dir)
    return train_file, eval_file, test_file
