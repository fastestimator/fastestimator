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
"""Download Caltech-UCSD Birds 200 dataset from http://www.vision.caltech.edu/visipedia/CUB-200.html."""
import os
import random
import tarfile
from pathlib import Path

import pandas as pd
import wget

from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(path=None):
    """Download the CUB200 dataset to local storage, if not already downloaded. This will generate a cub200.csv file,
    which contains all the path information.

    Args:
        path (str, optional): The path to store the CUB200 data. When `path` is not provided, will save at
            `fastestimator_data` under user's home directory.

    Returns:
        tuple: (csv_path, path) tuple, where

        * **csv_path** (str) -- Path to the summary csv file, containing the following columns:

            * image (str): Image directory relative to the returned path.
            * label (int): Image category
            * annotation (str): annotation file directory relative to the returned path.

        * **path** (str) -- Path to dataset root directory.

    """
    home = str(Path.home())

    if path is None:
        path = os.path.join(home, 'fastestimator_data', 'CUB200')
    else:
        path = os.path.join(os.path.abspath(path), 'CUB200')
    os.makedirs(path, exist_ok=True)

    csv_path = os.path.join(path, 'cub200.csv')
    image_compressed_path = os.path.join(path, 'images.tgz')
    annotation_compressed_path = os.path.join(path, 'annotations.tgz')
    image_extracted_path = os.path.join(path, 'images')
    annotation_extracted_path = os.path.join(path, 'annotations-mat')

    # download
    if not (os.path.exists(image_compressed_path) and os.path.exists(annotation_compressed_path)):
        print("Downloading data to {}".format(path))
        wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz', path, bar=bar_custom)
        wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz', path, bar=bar_custom)

    # extract
    if not (os.path.exists(image_extracted_path) and os.path.exists(annotation_extracted_path)):
        print("\nExtracting files ...")
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(path)
        with tarfile.open(annotation_compressed_path) as anno_tar:
            anno_tar.extractall(path)

    # glob and generate csv
    if not os.path.exists(csv_path):
        image_folder = os.path.join(path, "images")
        class_names = os.listdir(image_folder)
        label_map = {}
        images = []
        labels = []
        idx = 0
        for class_name in class_names:
            if not class_name.startswith("._"):
                image_folder_class = os.path.join(image_folder, class_name)
                label_map[class_name] = idx
                idx += 1
                image_names = os.listdir(image_folder_class)
                for image_name in image_names:
                    if not image_name.startswith("._"):
                        images.append(os.path.join(image_folder_class, image_name))
                        labels.append(label_map[class_name])
        zipped_list = list(zip(images, labels))
        random.shuffle(zipped_list)
        df = pd.DataFrame(zipped_list, columns=["image", "label"])
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, path))
        df['image'] = df['image'].apply(os.path.normpath)
        df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')
        df.to_csv(csv_path, index=False)
        print("Data summary is saved at {}".format(csv_path))
    return csv_path, path
