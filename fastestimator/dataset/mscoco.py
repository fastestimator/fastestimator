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
"""Download MS COCO 2014 dataset."""
import os
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget import bar_custom, callback_progress

wget.callback_progress = callback_progress
NUM_IMG_FILES = 82783


def _download_and_extract(url, filename, path):
    if not os.path.exists(os.path.join(path, filename)):
        print("Downloading data to {}".format(path))
        wget.download(url, path, bar=bar_custom)

    with zipfile.ZipFile(os.path.join(path, filename), 'r') as zip_file:
        print("\nExtracting {}".format(os.path.join(path, filename)))
        zip_file.extractall(path)


def _create_csv(data_path):
    filenames = glob(os.path.join(data_path, '*.jpg'))
    df = None
    if len(filenames) == NUM_IMG_FILES:
        df = pd.DataFrame()
        rel_filenames = [f.replace(data_path, '.') for f in filenames]
        df['image'] = rel_filenames
    else:
        print("One or more images are missing.")

    return df


def load_data(path=None):
    """Download the COCO dataset to local storage, if not already downloaded. This will generate a
    coco_train.csv file, which contains all the path information.

    Args:
        path (str, optional): The path to store the COCO data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        (tuple): tuple containing:
            csv_path (str): Path to the summary csv file.
            path (str): Path to data root directory.

    """
    home = str(Path.home())

    url = {'train': 'http://images.cocodataset.org/zips/train2014.zip'}
    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'MSCOCO')
    else:
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    csv_path = os.path.join(path, 'coco_train.csv')
    data_folder_path = os.path.join(path, 'train2014')

    if not os.path.exists(data_folder_path):
        _download_and_extract(url["train"], "train2014.zip", path)
        train_df = _create_csv(data_folder_path)
        train_df.to_csv(csv_path, index=False)
        print("Data summary is saved at {}".format(csv_path))
    else:
        if not len(glob(os.path.join(data_folder_path, '*.jpg'))) == NUM_IMG_FILES:
            print("One or more images are missing.")
            _download_and_extract(url["train"], "train2014.zip", path)
            train_df = _create_csv(data_folder_path)
            train_df.to_csv(csv_path, index=False)
            print("Data summary is saved at {}".format(csv_path))
        else:
            if not os.path.exists(csv_path):
                train_df = _create_csv(data_folder_path)
                train_df.to_csv(csv_path, index=False)
                print("Data summary is saved at {}".format(csv_path))
            else:
                print("Reusing existing dataset.")

    return csv_path, path
