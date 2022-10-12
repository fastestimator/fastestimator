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
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
from keras.datasets.cifar import load_batch

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.google_download_util import _download_file_from_google_drive


def load_data(root_dir: str = None,
              image_key: str = "x",
              label_key: str = "y",
              label_mode: str = "fine",
              channels_last: bool = True) -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the CIFAR100 dataset.

    Please consider using the ciFAIR100 dataset instead. CIFAR100 contains duplicates between its train and test sets.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        image_key: The key for image.
        label_key: The key for label.
        label_mode: Either "fine" for 100 classes or "coarse" for 20 classes.
        channels_last: Whether channel is last

    Returns:
        (train_data, eval_data)

    Raises:
        ValueError: If the label_mode is invalid.
    """
    print("\033[93m {}\033[00m".format("FastEstimator-Warn: Consider using the ciFAIR100 dataset instead."))
    if label_mode not in ['fine', 'coarse']:
        raise ValueError("label_mode must be one of either 'fine' or 'coarse'.")
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'cifar100')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'cifar100')
    os.makedirs(root_dir, exist_ok=True)

    image_compressed_path = os.path.join(root_dir, 'cifar100.tar.gz')
    image_extracted_path = os.path.join(root_dir, 'cifar-100-python')

    if not os.path.exists(image_extracted_path):
        if not os.path.exists(image_compressed_path):
            print("Downloading data to {}".format(root_dir))
            _download_file_from_google_drive('1ntXqOaXMaq4TcvpCaOCpqqNCjYy2oVsb', image_compressed_path)

        print("Extracting data to {}".format(root_dir))
        with tarfile.open(image_compressed_path) as img_tar:
            img_tar.extractall(root_dir)

    train_data_path = os.path.join(image_extracted_path, "train")
    x_train, y_train = load_batch(train_data_path, label_key=label_mode + "_labels")

    eval_data_path = os.path.join(image_extracted_path, "test")
    x_eval, y_eval = load_batch(eval_data_path, label_key=label_mode + "_labels")

    y_eval = np.array(y_eval)

    if channels_last:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_eval = x_eval.transpose(0, 2, 3, 1)

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    eval_data = NumpyDataset({image_key: x_eval, label_key: y_eval})
    return train_data, eval_data
