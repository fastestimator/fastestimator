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
"""Download The Street View House Numbers (SVHN) dataset."""
import os
import tarfile
from operator import add
from pathlib import Path

import h5py
import pandas as pd
import wget

from fastestimator.util.wget import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _get_name(index, hdf5_data):
    """Retrieves the image file name from hdf5 data for a specific index.

    Args:
        index (int): Index of image.
        hdf5_data (obj): h5py file containing bounding box information.

    Returns:
        (str): Image file name.

    """
    ref = hdf5_data['/digitStruct/name'][index, 0]
    file_name = ''.join([chr(item) for item in hdf5_data[ref][:]])
    return file_name


def _get_bbox(index, hdf5_data):
    """Retrieves the bounding box from hdf5 data for a specific index.

    Args:
        index (int): Index of image.
        hdf5_data (obj): h5py file containing bounding box information.

    Returns:
        (dict): Label and bounding box information including left, top, width, and height.

    """

    meta_data = {}
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def get_attrs(name, obj):
        vals = []
        if len(obj) == 1:
            vals.append(int(obj[()]))
        else:
            for h5_ref in obj[:, 0]:
                vals.append(int(hdf5_data[h5_ref][:]))

        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index, 0]
    hdf5_data[box].visititems(get_attrs)
    return meta_data


def _create_csv(data_folder, mode, csv_path):
    """Creates bounding boxes for all images. This will generate a csv file indicating for each image the label and
    bounding box coordinates and return the corresponding DataFrame.

    Args:
        data_folder (str): Path to data directory containing digitStruct.mat file.
        mode (str): Training or testing.
        csv_path (str): Path to save the csv file containing the bounding boxes information.

    """
    mat_data = h5py.File(os.path.join(data_folder, 'digitStruct.mat'), 'r')
    num_examples = len(mat_data['/digitStruct/bbox'])
    logging_interval = num_examples // 10
    print("Found {} examples for {}.".format(num_examples, mode))

    df = pd.DataFrame(columns=['image', 'label', 'x1', 'y1', 'x2', 'y2'])
    print("Retrieving bounding box for {} data. This will take several minutes ...".format(mode))
    for idx in range(num_examples):
        if idx % logging_interval == 0:
            print("{}%".format(int(idx / num_examples * 100)))
        img_name = _get_name(idx, mat_data)
        bbox = _get_bbox(idx, mat_data)
        row_dict = {
            'image': os.path.join(mode, img_name),
            'label': bbox["label"],
            'x1': bbox["left"],
            'y1': bbox["top"],
            'x2': list(map(add, bbox['left'], bbox['width'])),
            'y2': list(map(add, bbox['top'], bbox['height']))
        }
        df = df.append(row_dict, ignore_index=True)

    df.to_csv(csv_path, index=False)
    print("Data summary is saved at {}".format(csv_path))


def load_data(path=None):
    """Download the SVHN dataset to local storage, if not already downloaded. This will generate 2 csv files
    (train and test), which contain all the path information.

    Args:
        path (str, optional): The path to store the SVHN data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        (tuple): tuple containing:
            train_csv (str): Path to train csv file.
            test_csv (str): Path to test csv file.
            path (str): Path to data root directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'SVHN')
    else:
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')
    train_compressed_path = os.path.join(path, "train.tar.gz")
    test_compressed_path = os.path.join(path, "test.tar.gz")
    train_folder_path = os.path.join(path, "train")
    test_folder_path = os.path.join(path, "test")

    # download
    if not (os.path.exists(train_compressed_path) and os.path.exists(test_compressed_path)):
        print("Downloading data to {}".format(path))
        wget.download('http://ufldl.stanford.edu/housenumbers/train.tar.gz', path, bar=bar_custom)
        wget.download('http://ufldl.stanford.edu/housenumbers/test.tar.gz', path, bar=bar_custom)

    # extract
    if not (os.path.exists(train_folder_path) and os.path.exists(test_folder_path)):
        print("\nExtracting files ...")
        with tarfile.open(train_compressed_path) as tar:
            tar.extractall(path)
        with tarfile.open(test_compressed_path) as tar:
            tar.extractall(path)

    # glob and generate csv
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print("\nConstructing bounding box data ...")
        _create_csv(train_folder_path, "train", train_csv)
        _create_csv(test_folder_path, "test", test_csv)

    return train_csv, test_csv, path
