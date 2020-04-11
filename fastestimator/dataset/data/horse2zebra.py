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
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import wget

from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.dataset.dir_dataset import DirDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(batch_size: int, root_dir: Optional[str] = None) -> Tuple[BatchDataset, BatchDataset]:
    """Load and return the horse2zebra dataset.

    Sourced from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip. This method will
        download the data to local storage if the data has not been previously downloaded.

    Args:
        batch_size: The desired batch size.
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, eval_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'horse2zebra')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'horse2zebra')
    os.makedirs(root_dir, exist_ok=True)

    data_compressed_path = os.path.join(root_dir, 'horse2zebra.zip')
    data_folder_path = os.path.join(root_dir, 'images')

    if not os.path.exists(data_folder_path):
        # download
        if not os.path.exists(data_compressed_path):
            print("Downloading data to {}".format(root_dir))
            wget.download('https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                          root_dir,
                          bar=bar_custom)

        # extract
        print("\nExtracting files ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(root_dir)
        os.rename(os.path.join(root_dir, 'horse2zebra'), data_folder_path)

    test_a = DirDataset(root_dir=os.path.join(data_folder_path, 'testA'),
                        data_key="A",
                        file_extension='.jpg',
                        recursive_search=False)
    test_b = DirDataset(root_dir=os.path.join(data_folder_path, 'testB'),
                        data_key="B",
                        file_extension='.jpg',
                        recursive_search=False)
    train_a = DirDataset(root_dir=os.path.join(data_folder_path, 'trainA'),
                         data_key="A",
                         file_extension='.jpg',
                         recursive_search=False)
    train_b = DirDataset(root_dir=os.path.join(data_folder_path, 'trainB'),
                         data_key="B",
                         file_extension='.jpg',
                         recursive_search=False)
    outputs = (BatchDataset(datasets=[train_a, train_b], num_samples=[batch_size, batch_size]),
               BatchDataset(datasets=[test_a, test_b], num_samples=[batch_size, batch_size]))
    return outputs
