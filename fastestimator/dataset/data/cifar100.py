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
from typing import Tuple

import tensorflow as tf

from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data(image_key: str = "x",
              label_key: str = "y",
              label_mode: str = "fine") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the CIFAR100 dataset.

    Please consider using the ciFAIR100 dataset instead. CIFAR100 contains duplicates between its train and test sets.

    Args:
        image_key: The key for image.
        label_key: The key for label.
        label_mode: Either "fine" for 100 classes or "coarse" for 20 classes.

    Returns:
        (train_data, eval_data)

    Raises:
        ValueError: If the label_mode is invalid.
    """
    print("\033[93m {}\033[00m".format("FastEstimator-Warn: Consider using the ciFAIR100 dataset instead."))
    if label_mode not in ['fine', 'coarse']:
        raise ValueError("label_mode must be one of either 'fine' or 'coarse'.")
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    eval_data = NumpyDataset({image_key: x_eval, label_key: y_eval})
    return train_data, eval_data
