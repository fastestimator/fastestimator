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
from fastestimator.dataset.numpy_dataset import NumpyDataset
from sklearn import datasets


def load_data(image_key: str = "x", label_key: str = "y") -> NumpyDataset:
    """Load and return the Sklearn digits dataset.

    Args:
        image_key: The key for image.
        label_key: The key for label.

    Returns:
        (train_data, eval_data)
    """
    ds = datasets.load_digits()
    images = ds.images
    targets = ds.target
    return NumpyDataset({image_key: images, label_key: targets})
