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
import random
import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd
import wget

from fastestimator.dataset.csv_dataset import CSVDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None) -> CSVDataset:
    """Load and return the Caltech-UCSD Birds 200 (CUB200) dataset.

    Sourced from http://www.vision.caltech.edu/visipedia/CUB-200.html. This method will download the data to local
        storage if the data has not been previously downloaded.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        train_data
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'CUB200')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'CUB200')
    os.makedirs(root_dir, exist_ok=True)

    csv_path = os.path.join(root_dir, 'cub200.csv')
    image_compressed_path = os.path.join(root_dir, 'images.tgz')
    annotation_compressed_path = os.path.join(root_dir, 'annotations.tgz')
    image_extracted_path = os.path.join(root_dir, 'images')
    annotation_extracted_path = os.path.join(root_dir, 'annotations-mat')

    if not (os.path.exists(image_extracted_path) and os.path.exists(annotation_extracted_path)):
        # download
        if not (os.path.exists(image_compressed_path) and os.path.exists(annotation_compressed_path)):
            print("Downloading data to {}".format(root_dir))
            wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz', root_dir, bar=bar_custom)
            wget.download('http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz',
                          root_dir,
                          bar=bar_custom)

        # extract
        print("\nExtracting files ...")
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(root_dir)
        with tarfile.open(annotation_compressed_path) as anno_tar:
            anno_tar.extractall(root_dir)

    # glob and generate csv
    if not os.path.exists(csv_path):
        image_folder = os.path.join(root_dir, "images")
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
        df['image'] = df['image'].apply(lambda x: os.path.relpath(x, root_dir))
        df['image'] = df['image'].apply(os.path.normpath)
        df['annotation'] = df['image'].str.replace('images', 'annotations-mat').str.replace('jpg', 'mat')
        df.to_csv(csv_path, index=False)
        print("Data summary is saved at {}".format(csv_path))
    return CSVDataset(csv_path)
