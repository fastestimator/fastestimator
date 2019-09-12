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
"""
Download horse2zebra dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
"""
import os
import zipfile
import tempfile
from glob import glob

import pandas as pd
import wget


def _create_csv(img, key_name, parent_path, csv_name):
    df = pd.DataFrame()
    df[key_name] = [img_filename.replace(parent_path, '.') for img_filename in img]
    df.to_csv(csv_name, index=False)


def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "HORSE2ZEBRA")
    if not (os.path.exists(path)):
        os.makedirs(path)
    data_compressed_path = os.path.join(path, 'horse2zebra.zip')
    data_folder_path = os.path.join(path, 'horse2zebra')
    trainA_csv = os.path.join(path, 'trainA.csv')
    trainB_csv = os.path.join(path, 'trainB.csv')
    testA_csv = os.path.join(path, 'testA.csv')
    testB_csv = os.path.join(path, 'testB.csv')
    if not os.path.exists(data_compressed_path):
        print("Downloading data to {}:".format(path))
        wget.download('https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip', path)
    if not os.path.exists(data_folder_path):
        print('\nExtracting files...')
        with zipfile.ZipFile(data_compressed_path, 'r') as zip_file:
            zip_file.extractall(path)
    if not os.path.exists(trainA_csv):
        trainA_img = glob(os.path.join(path, 'horse2zebra', 'trainA', '*.jpg'))
        _create_csv(trainA_img, 'imgA', path, trainA_csv)
    if not os.path.exists(trainB_csv):
        trainB_img = glob(os.path.join(path, 'horse2zebra', 'trainB', '*.jpg'))
        _create_csv(trainB_img, 'imgB', path, trainB_csv)
    if not os.path.exists(testA_csv):
        testA_img = glob(os.path.join(path, 'horse2zebra', 'testA', '*.jpg'))
        _create_csv(testA_img, 'imgA', path, testA_csv)
    if not os.path.exists(testB_csv):
        testB_img = glob(os.path.join(path, 'horse2zebra', 'testB', '*.jpg'))
        _create_csv(testB_img, 'imgB', path, testB_csv)
    return trainA_csv, trainB_csv, testA_csv, testB_csv, path
