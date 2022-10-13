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
from typing import Tuple

import numpy as np

from fastestimator.dataset.data.cifair10 import _load_batch
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.google_download_util import _download_file_from_google_drive


def load_data(root_dir: str = None, image_key: str = "x", label_key: str = "y",
              label_mode: str = "fine") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the ciFAIR100 dataset.

    This is the cifar100 dataset but with test set duplicates removed and replaced. See
    https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
    dataset.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, test_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'ciFAIR100')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'ciFAIR100')
    os.makedirs(root_dir, exist_ok=True)

    image_compressed_path = os.path.join(root_dir, 'ciFAIR100.zip')
    image_extracted_path = os.path.join(root_dir, 'ciFAIR-100')

    if not os.path.exists(image_extracted_path):
        if not os.path.exists(image_compressed_path):
            print("Downloading data to {}".format(root_dir))
            _download_file_from_google_drive('1ZE_wf5UTd9fJqBgikb7MJtfFEeAfXybS', image_compressed_path)

        print("Extracting data to {}".format(root_dir))
        shutil.unpack_archive(image_compressed_path, root_dir)

    fpath = os.path.join(image_extracted_path, 'train')
    x_train, y_train = _load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(image_extracted_path, 'test')
    x_test, y_test = _load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.array(y_train, dtype=np.uint8)
    y_test = np.array(y_test, dtype=np.uint8)

    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    x_test = x_test.astype(x_train.dtype)

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    test_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, test_data
