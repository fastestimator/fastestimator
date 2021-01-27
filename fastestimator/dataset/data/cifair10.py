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
import pickle
from typing import Tuple, List

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the ciFAIR10 dataset.

    This is the cifar10 dataset but with test set duplicates removed and replaced. See
    https://arxiv.org/pdf/1902.00423.pdf or https://cvjena.github.io/cifair/ for details. Cite the paper if you use the
    dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, test_data)
    """
    dirname = 'ciFAIR-10'
    archive_name = 'ciFAIR-10.zip'
    origin = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    md5_hash = 'ca08fd390f0839693d3fc45c4e49585f'

    path = get_file(archive_name, origin=origin, file_hash=md5_hash, hash_algorithm='md5', extract=True,
                    archive_format='zip')
    path = os.path.join(os.path.dirname(path), dirname)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, f'data_batch_{i}')
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = _load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    test_data = NumpyDataset({image_key: x_test, label_key: y_test})
    return train_data, test_data


def _load_batch(file_path: str, label_key: str = 'labels') -> Tuple[np.ndarray, List[int]]:
    """An internal utility for parsing ciFAIR data.

    Args:
        file_path: The path to the file to parse.
        label_key: Key used for the label data within the data dictionary.

    Returns:
        (data, labels)
    """
    with open(file_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        # decode bytes to utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf-8') if isinstance(k, bytes) else k] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
