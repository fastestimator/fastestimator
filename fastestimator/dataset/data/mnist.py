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
from pathlib import Path
from typing import Tuple

import numpy as np
from torchvision import datasets

from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data(image_key: str = "x", label_key: str = "y", root_dir: str = None) -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the MNIST dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, eval_data)
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'mnist')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'mnist')

    train_data = datasets.MNIST(root_dir, train=True, download=True, transform=None)
    eval_data = datasets.MNIST(root_dir, train=False, transform=None)

    x_train = np.array([image for image, _ in train_data]).astype(np.float32)
    y_train = np.array([clas for _, clas in train_data])

    x_eval = np.array([image for image, _ in eval_data]).astype(np.float32)
    y_eval = np.array([clas for _, clas in eval_data])

    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    eval_data = NumpyDataset({image_key: x_eval, label_key: y_eval})
    return train_data, eval_data
