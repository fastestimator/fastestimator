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
"""Download horse2zebra dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip.
"""
import os
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _create_csv(img_path, key_name, parent_path, csv_path):
    df = pd.DataFrame()
    df[key_name] = img_path
    df[key_name] = df[key_name].apply(lambda x: os.path.relpath(x, parent_path))
    df[key_name] = df[key_name].apply(os.path.normpath)
    df.to_csv(csv_path, index=False)
    print("Data summary is saved at {}".format(csv_path))


def load_data(path=None):
    """Download the horse2zebra dataset to local storage, if not already downloaded. This will generate 4 csv files
    (trainA, trainB, testA, testB), which contain all the path information.

    Args:
        path (str, optional): The path to store the horse2zebra data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (train_a_csv, train_b_csv, test_a_csv, test_b_csv, path) tuple, where
        
        * **train_a_csv** (str) -- Path to trainA csv file, containing the following column:
        
            * imgA: Image directory relative to the returned path.
            
        * **train_b_csv** (str) -- Path to trainB csv file, containing the following column:
        
            * imgB: Image directory relative to the returned path.
            
        * **test_a_csv** (str) -- Path to testA csv file, containing the following column:
        
            * imgA: Image directory relative to the returned path.
            
        * **test_b_csv** (str) -- Path to testB csv file, containing the following column:
        
            * imgB: Image directory relative to the returned path.
            
        * **path** (str) -- Path to data directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'horse2zebra')
    else:
        path = os.path.join(os.path.abspath(path), 'horse2zebra')
    os.makedirs(path, exist_ok=True)

    data_compressed_path = os.path.join(path, 'horse2zebra.zip')
    data_folder_path = os.path.join(path, 'images')
    train_a_csv = os.path.join(path, 'trainA.csv')
    train_b_csv = os.path.join(path, 'trainB.csv')
    test_a_csv = os.path.join(path, 'testA.csv')
    test_b_csv = os.path.join(path, 'testB.csv')

    # download
    if not os.path.exists(data_compressed_path):
        print("Downloading data to {}".format(path))
        wget.download('https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                      path,
                      bar=bar_custom)

    # extract
    if not os.path.exists(data_folder_path):
        print("\nExtracting files ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)
        os.rename(os.path.join(path, 'horse2zebra'), os.path.join(path, 'images'))

    # glob and generate csv
    if not os.path.exists(train_a_csv):
        train_a_img = glob(os.path.join(data_folder_path, 'trainA', '*.jpg'))
        _create_csv(train_a_img, 'imgA', path, train_a_csv)
    if not os.path.exists(train_b_csv):
        train_b_img = glob(os.path.join(data_folder_path, 'trainB', '*.jpg'))
        _create_csv(train_b_img, 'imgB', path, train_b_csv)
    if not os.path.exists(test_a_csv):
        test_a_img = glob(os.path.join(data_folder_path, 'testA', '*.jpg'))
        _create_csv(test_a_img, 'imgA', path, test_a_csv)
    if not os.path.exists(test_b_csv):
        test_b_img = glob(os.path.join(data_folder_path, 'testB', '*.jpg'))
        _create_csv(test_b_img, 'imgB', path, test_b_csv)
    return train_a_csv, train_b_csv, test_a_csv, test_b_csv, path
