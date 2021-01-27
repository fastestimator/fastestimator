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
from typing import Tuple

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from fastestimator.dataset.data.cifair10 import _load_batch
from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data(image_key: str = "x",
              label_key: str = "y",
              label_mode: str = "fine") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the ciFAIR100 dataset.

    This is the cifar100 dataset but with test set duplicates removed and replaced. See
    https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
    dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.
        label_mode: Either "fine" for 100 classes or "coarse" for 20 classes.

    Returns:
        (train_data, test_data)

    Raises:
        ValueError: If the label_mode is invalid.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError("label_mode must be one of either 'fine' or 'coarse'.")

    dirname = 'ciFAIR-100'
    archive_name = 'ciFAIR-100.zip'
    origin = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip'
    md5_hash = 'ddc236ab4b12eeb8b20b952614861a33'

    path = get_file(archive_name, origin=origin, file_hash=md5_hash, hash_algorithm='md5', extract=True,
                    archive_format='zip')
    path = os.path.join(os.path.dirname(path), dirname)

    fpath = os.path.join(path, 'train')
    x_train, y_train = _load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = _load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    test_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, test_data
