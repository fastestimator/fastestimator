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

from fastestimator.dataset.labeled_dir_dataset import LabeledDirDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None) -> Tuple[LabeledDirDataset, LabeledDirDataset]:
    """Load and return the Tiny ImageNet dataset.
    Sourced from http://cs231n.stanford.edu/tiny-imagenet-200.zip. This method will
        download the data to local storage if the data has not been previously downloaded.
    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
    Returns:
        (train_data, eval_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'tiny_imagenet')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'tiny_imagenet')
    os.makedirs(root_dir, exist_ok=True)

    data_compressed_path = os.path.join(root_dir, 'tiny-imagenet-200.zip')
    train_file_path = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
    val_file_path = os.path.join(root_dir, 'tiny-imagenet-200', 'val')

    if (os.path.exists(train_file_path) == False) or (os.path.exists(val_file_path) == False):
        # download
        if not os.path.exists(data_compressed_path):
            print("Downloading data to {}".format(root_dir))
            wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root_dir, bar=bar_custom)

        # extract
        print("\nExtracting files ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(root_dir)
        #os.rename(os.path.join(root_dir, 'horse2zebra'), data_folder_path)

        current_dir = os.path.join(root_dir, 'tiny-imagenet-200')

        # Update Train Directory
        for root, _, files in os.walk(os.path.join(current_dir, 'train')):
            for filename in files:
                #print(os.path.join(root, filename))
                if filename.endswith('.txt'):
                    os.remove(os.path.join(root, filename))
                else:
                    p = Path(os.path.join(root, filename)).absolute()
                    parent_dir = p.parents[1]
                    p.rename(parent_dir / p.name)
            if len(os.listdir(root)) == 0:
                os.rmdir(root)

        # Update Val Directory
        for line in open(os.path.join(current_dir, 'val', 'val_annotations.txt')).readlines():
            file_data = [n for n in line.split()]
            folder_path = os.path.join(current_dir, 'val', file_data[1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(current_dir, 'val', 'images', file_data[0])
            os.rename(file_path, os.path.join(folder_path, file_data[0]))
        os.rmdir(os.path.join(current_dir, 'val', 'images'))
        os.remove(os.path.join(current_dir, 'val', 'val_annotations.txt'))

    root_path = os.path.join(root_dir, 'tiny-imagenet-200')
    train_outputs = LabeledDirDataset(os.path.join(root_path, "train"), data_key='image', label_key='label')

    eval_outputs = LabeledDirDataset(os.path.join(root_path, "val"), data_key='image', label_key='label')

    return (train_outputs, eval_outputs)
