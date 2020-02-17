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
""" Mendeley dataset API.
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
Images for Classification", Mendeley Data, v2 http://dx.doi.org/10.17632/rscbjbr9sj.2

CC BY 4.0 licence:
https://creativecommons.org/licenses/by/4.0/
"""

import os
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd

import requests
from tqdm import tqdm


def _create_csv(img_path, parent_path, csv_path):
    df = pd.DataFrame()
    df['image'] = img_path
    df['image'] = df['image'].apply(lambda x: os.path.relpath(x, parent_path))
    df['image'] = df['image'].apply(os.path.normpath)
    df['label'] = df['image'].str.contains('PNEUMONIA').astype(int)
    df.to_csv(csv_path, index=False)
    print("Data summary is saved at {}".format(csv_path))


def load_data(path=None):
    """Download the Mendeley dataset to local storage, if not already downloaded. This will generate 2 csv files
    (train, test), which contain all the path information.

    Args:
        path (str, optional): The path to store the Mendeley data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (train_csv, test_csv, path) tuple, where
        
        * **train_csv** (str) -- Path to train csv file, containing the following columns:
        
            * image (str): Image directory relative to the returned path.
            * label (int): Label which indicates the presence of pneumonia (1 for positive, 0 for negative).
            
        * **test_csv** (str) -- Path to test csv file, containing same columns as train_csv.
        
        * **path** (str) -- Path to data directory.

    """
    url = 'https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/' \
        'ChestXRay2017.zip?dl=1'
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'Mendeley')
    else:
        path = os.path.join(os.path.abspath(path), 'Mendeley')
    os.makedirs(path, exist_ok=True)

    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')
    data_compressed_path = os.path.join(path, 'ChestXRay2017.zip')
    data_folder_path = os.path.join(path, 'chest_xray')

    # download
    if not os.path.exists(data_compressed_path):
        print("Downloading data to {}".format(path))
        stream = requests.get(url, stream=True)  # python wget does not work
        total_size = int(stream.headers.get('content-length', 0))
        block_size = int(1e6)  # 1 MB
        progress = tqdm(total=total_size, unit='B', unit_scale=True)
        with open(data_compressed_path, 'wb') as outfile:
            for data in stream.iter_content(block_size):
                progress.update(len(data))
                outfile.write(data)
        progress.close()

    # extract
    if not os.path.exists(data_folder_path):
        print("\nExtracting file ...")
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(path=path)

    # glob and generate csv
    if not os.path.exists(train_csv):
        train_img = glob(os.path.join(data_folder_path, 'train', '**', '*.jpeg'))
        _create_csv(train_img, path, train_csv)
    if not os.path.exists(test_csv):
        test_img = glob(os.path.join(data_folder_path, 'test', '**', '*.jpeg'))
        _create_csv(test_img, path, test_csv)

    return train_csv, test_csv, path
