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
from typing import Dict, List, Optional, Tuple

import pandas as pd
import wget

from fastestimator.dataset.csv_dataset import CSVDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def _create_csv(images: List[str], label_dict: Dict[str, int], csv_path: str) -> None:
    """A helper function to create and save csv files.

    Args:
        images: List of image id's.
        label_dict: Mapping of class name to label id.
        csv_path: Path to save the csv file.
    """

    df = pd.DataFrame(images, columns=["image"])
    df["label"] = df["image"].apply(lambda x: label_dict[x.split("/")[0]])
    df["image"] = "food-101/images/" + df["image"] + ".jpg"
    df.to_csv(csv_path, index=False)
    return None


def load_data(root_dir: Optional[str] = None) -> Tuple[CSVDataset, CSVDataset]:
    """Load and return the Food-101 dataset.

    Food-101 dataset is a collection of images from 101 food categories.
    Sourced from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, test_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'Food_101')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'Food_101')
    os.makedirs(root_dir, exist_ok=True)

    image_compressed_path = os.path.join(root_dir, 'food-101.tar.gz')
    image_extracted_path = os.path.join(root_dir, 'food-101')

    train_csv_path = os.path.join(root_dir, 'train.csv')
    test_csv_path = os.path.join(root_dir, 'test.csv')

    if not os.path.exists(image_extracted_path):
        # download
        if not os.path.exists(image_compressed_path):
            print("Downloading data to {}".format(root_dir))
            wget.download('http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz', root_dir, bar=bar_custom)

        # extract
        print("\nExtracting files ...")
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(root_dir)

    labels = open(os.path.join(root_dir, "food-101/meta/classes.txt"), "r").read().split()
    label_dict = {labels[i]: i for i in range(len(labels))}

    if not os.path.exists(train_csv_path):
        train_images = open(os.path.join(root_dir, "food-101/meta/train.txt"), "r").read().split()
        random.shuffle(train_images)
        _create_csv(train_images, label_dict, train_csv_path)
    if not os.path.exists(test_csv_path):
        test_images = open(os.path.join(root_dir, "food-101/meta/test.txt"), "r").read().split()
        random.shuffle(test_images)
        _create_csv(test_images, label_dict, test_csv_path)

    return CSVDataset(train_csv_path), CSVDataset(test_csv_path)
