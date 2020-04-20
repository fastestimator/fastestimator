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


def load_data(image_key: str = "x", label_key: str = "y") -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the MNIST dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, eval_data)
    """
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    train_data = NumpyDataset({image_key: x_train, label_key: y_train})
    eval_data = NumpyDataset({image_key: x_eval, label_key: y_eval})
    return train_data, eval_data
