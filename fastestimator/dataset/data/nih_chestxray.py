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
import tarfile
from pathlib import Path
from typing import Optional

import wget

from fastestimator.dataset.dir_dataset import DirDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _download_data(link: str, data_path: str, idx: int, total_idx: int) -> None:
    """A helper function to run wget.

    Args:
        link: The download link.
        data_path: Where to save the downloaded data.
        idx: The current download index.
        total_idx: How many files total will be downloaded.
    """
    if not os.path.exists(data_path):
        print("Downloading data to {}, file: {} / {}".format(data_path, idx + 1, total_idx))
        wget.download(link, data_path, bar=bar_custom)


def load_data(root_dir: Optional[str] = None) -> DirDataset:
    """Load and return the NIH Chest X-ray dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        train_data
    """
    if root_dir is None:
        root_dir = os.path.join(str(Path.home()), 'fastestimator_data', 'NIH_Chestxray')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'NIH_Chestxray')
    os.makedirs(root_dir, exist_ok=True)

    image_extracted_path = os.path.join(root_dir, 'images')

    if not os.path.exists(image_extracted_path):
        # download data
        links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]
        data_paths = [os.path.join(root_dir, "images_{}.tar.gz".format(x)) for x in range(len(links))]
        for idx, (link, data_path) in enumerate(zip(links, data_paths)):
            _download_data(link, data_path, idx, len(links))

        # extract data
        for idx, data_path in enumerate(data_paths):
            print("Extracting {}, file {} / {}".format(data_path, idx + 1, len(links)))
            with tarfile.open(data_path) as img_tar:
                img_tar.extractall(root_dir)

    return DirDataset(image_extracted_path, file_extension='.png', recursive_search=False)
