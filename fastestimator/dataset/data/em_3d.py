# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.google_download_util import _download_file_from_google_drive

color_mapping = {
    0: [0, 0, 0, 0],
    1: [0, 40, 255, 255],
    2: [0, 212, 255, 255],
    3: [124, 255, 121, 255],
    4: [127, 0, 0, 255],
    5: [255, 70, 0, 255],
    6: [255, 229, 0, 255]
}


def generate_tiles(input_image, tile_size=256, overlap=128):
    """
        This method to crop the image into smaller crops.

        Args:
            input_image: numpy array of input image
            tile_size: The crop size
            overlap: Overlap between two crops

        Returns:
            tiles: numpy array of image crops
    """
    stride = tile_size - overlap
    _, height, width = input_image.shape
    tiles = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if i + tile_size < height and j + tile_size < width:
                tiles.append(input_image[:, i:i + tile_size, j:j + tile_size])
    return np.array(tiles)


def get_encode_label(label_data):
    """
        One hot encode the input mask

        Args:
            label_data: Color encoded input label

        Returns:
            encoded_label: one hot encoded label
    """
    encoded_label = np.zeros(label_data.shape[:-1], np.uint8)
    for i, value in color_mapping.items():
        encoded_label[np.all(label_data == value, axis=-1)] = i
    return encoded_label


def load_data(root_dir: Optional[str] = None, image_key: str = "image",
              label_key: str = "label") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the 3d electron microscope platelet dataset.


    Sourced from https://bio3d-vision.github.io/platelet-description.
    Electronic Microscopy 3D cell dataset, consists of 2 3D images, one 800x800x50 and the other 800x800x24.
    The 800x800x50 is used as training dataset and 800x800x24 is used for validation. Instead of using the entire
    800x800 images, the 800x800x50 is tiled into 256x256x24 tiles with an overlap of 128 producing around 75 training
    images and similarly the 800x800x24 image is tiled to produce 25 validation images.

    The method downloads the dataset from google drive and provides train and validation NumpyDataset.
    While the dataset contains encoded value 0 as background, its omitted in the one hot encoded class label provided
    by this method. Below indexes represent the labels in channel layer.
        Index	Class name
        0		Cell
        1		Mitochondria
        2		Alpha granule
        3		Canalicular vessel
        4		Dense granule body
        5		Dense granule core

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided,
            the data will be saved into `fastestimator_data` under the user's home directory.
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, eval_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'electronmicroscopy')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'electronmicroscopy')
    os.makedirs(root_dir, exist_ok=True)

    data_compressed_path = os.path.join(root_dir, 'images_and_labels_rgba.zip')
    data_folder_path = os.path.join(root_dir, 'platelet-em/images')

    if not os.path.exists(data_folder_path):
        _download_file_from_google_drive('1OMVY1bkfssYdH11xuFdxv7nhqEjzCY7L', data_compressed_path)
        shutil.unpack_archive(data_compressed_path, root_dir)

    train_data = tifffile.imread(os.path.join(root_dir, 'platelet-em/images/50-images.tif'))
    val_data = tifffile.imread(os.path.join(root_dir, 'platelet-em/images/24-images.tif'))
    train_label_data = tifffile.imread(os.path.join(root_dir, 'platelet-em/labels-semantic/50-semantic.tif'))
    val_label_data = tifffile.imread(os.path.join(root_dir, 'platelet-em/labels-semantic/24-semantic.tif'))

    encoded_train_label = get_encode_label(train_label_data)
    encoded_val_label = get_encode_label(val_label_data)

    train_data_slices = [train_data[0:24], train_data[13:37], train_data[26:50]]
    train_label_slices = [encoded_train_label[0:24], encoded_train_label[13:37], encoded_train_label[26:50]]

    training_data_tiles = np.moveaxis(
        np.concatenate([generate_tiles(slice_data) for slice_data in train_data_slices], axis=0), 1, -1)
    training_label_tiles = np.moveaxis(
        np.concatenate([generate_tiles(slice_data) for slice_data in train_label_slices], axis=0), 1, -1)

    val_data_tiles = np.moveaxis(generate_tiles(val_data), 1, -1)
    val_label_tiles = np.moveaxis(generate_tiles(encoded_val_label), 1, -1)

    val_label_tiles = np.eye(7)[val_label_tiles].take(indices=range(1, 7), axis=-1)
    training_label_tiles = np.eye(7)[training_label_tiles].take(indices=range(1, 7), axis=-1)

    train_data = NumpyDataset({image_key: training_data_tiles, label_key: training_label_tiles})
    eval_data = NumpyDataset({image_key: val_data_tiles, label_key: val_label_tiles})

    return train_data, eval_data
