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
"""Mendeley dataset API.

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018),
“Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”,
Mendeley Data, v2
http://dx.doi.org/10.17632/rscbjbr9sj.2

CC BY 4.0 licence:
http://creativecommons.org/licenses/by/4.0
"""
import os
import tempfile
import zipfile

import pandas as pd
import wget


def img_data_constructor(data_folder, mode, csv_path):
    normal_cases_dir = os.path.join(data_folder, 'NORMAL')
    pneumonia_cases_dir = os.path.join(data_folder, 'PNEUMONIA')

    data = []

    for dirpath, _, filenames in os.walk(normal_cases_dir):
        for f in filenames:
            # Select only jpeg files:
            if f[-4:] == 'jpeg':
                img = os.path.abspath(os.path.join(dirpath, f))
                data.append((img, 0))

    for dirpath, _, filenames in os.walk(pneumonia_cases_dir):
        for f in filenames:
            if f[-4:] == 'jpeg':
                img = os.path.abspath(os.path.join(dirpath, f))
                data.append((img, 1))

    num_example = len(data)
    print("found %d number of examples for %s" % (num_example, mode))

    data = pd.DataFrame(data, columns=['x', 'y'], index=None)
    data = data.sample(frac=1.).reset_index(drop=True)
    data.to_csv(csv_path, index=False)

    return data


def load_data(path=None):
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "Mendeley")
    if not os.path.exists(path):
        os.mkdir(path)

    train_csv = os.path.join(path, "train_data.csv")
    test_csv = os.path.join(path, "test_data.csv")

    if not os.path.exists(os.path.join(path, "ChestXRay2017.zip")):
        print("downloading data to %s" % path)
        wget.download('https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/ChestXRay2017.zip', path)

    if not (os.path.exists(os.path.join(path, "chest_xray/train")) and os.path.exists(os.path.join(path, "chest_xray/test"))):
        print(" ")
        print("extracting data...")
        with zipfile.ZipFile(os.path.join(path, "ChestXRay2017.zip"), 'r') as zip_file:
            zip_file.extractall(path=path)

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print("constructing data for FastEstimator...")

        train_folder = os.path.join(path, "chest_xray/train")
        test_folder = os.path.join(path, "chest_xray/test")

        img_data_constructor(train_folder, "train", train_csv)
        img_data_constructor(test_folder, "test", test_csv)

    return train_csv, test_csv, path
