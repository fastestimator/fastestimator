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
from typing import Dict, List, Optional, Tuple

import h5py
import pandas as pd
import tqdm
import wget

from fastestimator.dataset.pickle_dataset import PickleDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _get_name(index: int, hdf5_data: h5py.File) -> str:
    """Retrieves the image file name from hdf5 data for a specific index.

    Args:
        index: Index of the image.
        hdf5_data: h5py file containing bounding box information.

    Returns:
        Image file name.
    """
    ref = hdf5_data['/digitStruct/name'][index, 0]
    file_name = ''.join([chr(item) for item in hdf5_data[ref][:]])
    return file_name


def _get_bbox(index: int, hdf5_data: h5py.File) -> Dict[str, List[int]]:
    """Retrieves the bounding box from hdf5 data for a specific index.

    Args:
        index: Index of image.
        hdf5_data: h5py file containing bounding box information.

    Returns:
        Label and bounding box information including left, top, width, and height.
    """
    meta_data = {'height': [], 'label': [], 'left': [], 'top': [], 'width': []}

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


def _extract_metadata(data_folder: str, mode: str, save_path: str) -> None:
    """Creates bounding boxes for all images.

    This will generate a file indicating for each image the label and bounding box coordinates.

    Args:
        data_folder: Path to data directory containing digitStruct.mat file.
        mode: "train" or "test".
        save_path: Path to save the file containing the bounding box information.
    """
    mat_path = os.path.join(data_folder, 'digitStruct.mat')
    mat_data = h5py.File(mat_path, 'r')
    num_examples = len(mat_data['/digitStruct/bbox'])
    mat_data.close()
    print("Found {} examples for {}.".format(num_examples, mode))

    df = pd.DataFrame(columns=['image', 'label', 'x1', 'y1', 'width', 'height'])
    print("Retrieving bounding box for {} data. This will take several minutes ...".format(mode))

    with h5py.File(mat_path, 'r') as hdf5_data:
        for index in tqdm.trange(num_examples, desc="Gathering Boxes"):
            img_name = _get_name(index, hdf5_data)
            bbox = _get_bbox(index, hdf5_data)
            row_dict = {
                'image': os.path.join(mode, img_name),
                'label': bbox["label"],
                'x1': bbox["left"],
                'y1': bbox["top"],
                'width': bbox['width'],
                'height': bbox['height']
            }
            df = df.append(row_dict, ignore_index=True)

    df.to_pickle(save_path)
    print("Data summary is saved at {}".format(save_path))


def load_data(root_dir: Optional[str] = None) -> Tuple[PickleDataset, PickleDataset]:
    """Load and return the Street View House Numbers (SVHN) dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, test_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'SVHN')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'SVHN')
    os.makedirs(root_dir, exist_ok=True)

    train_file_path = os.path.join(root_dir, 'train.pickle')
    test_file_path = os.path.join(root_dir, 'test.pickle')
    train_compressed_path = os.path.join(root_dir, "train.tar.gz")
    test_compressed_path = os.path.join(root_dir, "test.tar.gz")
    train_folder_path = os.path.join(root_dir, "train")
    test_folder_path = os.path.join(root_dir, "test")

    if not os.path.exists(train_folder_path):
        # download
        if not os.path.exists(train_compressed_path):
            print("Downloading train data to {}".format(root_dir))
            wget.download('http://ufldl.stanford.edu/housenumbers/train.tar.gz', root_dir, bar=bar_custom)
        # extract
        print("\nExtracting files ...")
        with tarfile.open(train_compressed_path) as tar:
            tar.extractall(root_dir)

    if not os.path.exists(test_folder_path):
        # download
        if not os.path.exists(test_compressed_path):
            print("Downloading eval data to {}".format(root_dir))
            wget.download('http://ufldl.stanford.edu/housenumbers/test.tar.gz', root_dir, bar=bar_custom)
        # extract
        print("\nExtracting files ...")
        with tarfile.open(test_compressed_path) as tar:
            tar.extractall(root_dir)

    # glob and generate bbox files
    if not os.path.exists(train_file_path):
        print("\nConstructing bounding box data ...")
        _extract_metadata(train_folder_path, "train", train_file_path)
    if not os.path.exists(test_file_path):
        print("\nConstructing bounding box data ...")
        _extract_metadata(test_folder_path, "test", test_file_path)

    return PickleDataset(train_file_path), PickleDataset(test_file_path)
