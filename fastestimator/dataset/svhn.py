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
import tempfile
from op import add

import h5py
import pandas as pd
import wget


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    """Retrieves the bounding box from hdf5 data for a specific index.
    
    Args:
        index (int): index of image.
        hdf5_data (h5py file): h5py file containing bounding box information.
    
    Returns:
        dictionnary: label, left, top, width and height values for the bounding box.
    """
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [int(hdf5_data[attr.value[i].item()].value[0][0])
                  for i in range(len(attr))] if len(attr) > 1 else [int(attr.value[0][0])]
        attrs[key] = values
    return attrs


def img_boundingbox_data_constructor(data_folder, mode, csv_path):
    """Creates bounding boxes for all images. This will generate a csv file indicating for each image the label and bounding box coordinates 
    and return the corresponding DataFrame.
    
    Args:
        data_folder (string): path to data directory containing digitStruct.mat file.
        mode (string): training or testing.
        csv_path (string): path to save the csv file containing the bounding boxes information.
    
    Returns:
        DataFrame: bounding boxes information (image, label and coordinates)
    """
    f = h5py.File(os.path.join(data_folder, "digitStruct.mat"), 'r')
    row_list = []
    num_example = f['/digitStruct/bbox'].shape[0]
    logging_interval = num_example // 10
    print("found %d number of examples for %s" % (num_example, mode))
    for j in range(num_example):
        if j % logging_interval == 0:
            print("retrieving bounding box for %s: %.2f%%" % (mode, j / num_example * 100))
        img_name = get_name(j, f)
        bbox = get_bbox(j, f)
        row_dict = {
            'image': os.path.join(mode, img_name),
            'label': bbox["label"],
            'x1': bbox["left"],
            'y1': bbox["top"],
            'x2': list(map(add, bbox["left"], bbox["width"])),
            'y2': list(map(add, bbox["top"], bbox["height"]))
        }
        row_list.append(row_dict)
    bbox_df = pd.DataFrame(row_list, columns=['image', 'label', 'x1', 'y1', 'x2', 'y2'])
    bbox_df.to_csv(csv_path, index=False)
    return bbox_df


def load_data(path=None):
    """Downloads the svhn dataset to local storage, if not already downloaded. This will generate 2 csv files (train and test), which contain all the path
        information.

    Args:
        path (str, optional): The path to store the svhn data. Defaults to None, will save at `tempfile.gettempdir()`.

    Returns:
    string: path to train csv file.
    string: path to test csv file.
    string: path to data directory.
    """
    if path is None:
        path = os.path.join(tempfile.gettempdir(), ".fe", "SVHN")
    if not os.path.exists(path):
        os.makedirs(path)
    train_csv = os.path.join(path, "train_data.csv")
    test_csv = os.path.join(path, "test_data.csv")
    train_compressed_path = os.path.join(path, "train.tar.gz")
    test_compressed_path = os.path.join(path, "test.tar.gz")
    train_folder_path = os.path.join(path, "train")
    test_folder_path = os.path.join(path, "test")
    if not (os.path.exists(train_compressed_path) and os.path.exists(test_compressed_path)):
        print("Downloading data to %s" % path)
        wget.download('http://ufldl.stanford.edu/housenumbers/train.tar.gz', path)
        wget.download("http://ufldl.stanford.edu/housenumbers/test.tar.gz", path)
    if not (os.path.exists(train_folder_path) and os.path.exists(test_folder_path)):
        print(" ")
        print("extracting data...")
        with tarfile.open(train_compressed_path) as tar:
            tar.extractall(path)
        with tarfile.open(test_compressed_path) as tar:
            tar.extractall(path)
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print("constructing bounding box data...")
        img_boundingbox_data_constructor(train_folder_path, "train", train_csv)
        img_boundingbox_data_constructor(test_folder_path, "test", test_csv)
    return train_csv, test_csv, path
