# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wget
from scipy.io import loadmat

from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.util.wget_util import bar_custom, callback_progress

wget.callback_progress = callback_progress


def load_data(root_dir: Optional[str] = None) -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the SVHN Cropped digits dataset.

    For more information about this dataset please visit http://ufldl.stanford.edu/housenumbers/. Here, we are using
    Format 2 to get MNIST-like 32-by-32 images.

    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.

    Returns:
        (train_data, test_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'SVHN_Cropped')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'SVHN_Cropped')
    os.makedirs(root_dir, exist_ok=True)

    # download data to memory
    train_path = os.path.join(root_dir, "train_32x32.mat")
    test_path = os.path.join(root_dir, "test_32x32.mat")

    if not os.path.exists(train_path):
        print("Downloading train data to {}".format(root_dir))
        wget.download('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', root_dir, bar=bar_custom)

    if not os.path.exists(test_path):
        print("Downloading test data to {}".format(root_dir))
        wget.download('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', root_dir, bar=bar_custom)

    xy_train = loadmat(train_path)
    xy_test = loadmat(test_path)

    # setting label of '0' digit from '10' to '0'
    xy_train['y'][xy_train['y'] == 10] = 0
    xy_test['y'][xy_test['y'] == 10] = 0

    # make datasets
    train_data = NumpyDataset({"x": np.transpose(xy_train['X'], (3, 0, 1, 2)), "y": xy_train['y']})
    test_data = NumpyDataset({"x": np.transpose(xy_test['X'], (3, 0, 1, 2)), "y": xy_test['y']})

    return train_data, test_data
