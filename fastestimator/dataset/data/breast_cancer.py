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

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data() -> Tuple[NumpyDataset, NumpyDataset]:
    """Load and return the UCI ML Breast Cancer Wisconsin (Diagnostic) dataset.

    For more information about this dataset and the meaning of the features it contains, see the sklearn documentation.

    Returns:
        (train_data, eval_data)
    """
    (x, y) = load_breast_cancer(return_X_y=True)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_eval = np.float32(x_train), np.float32(x_eval)
    train_data = NumpyDataset({"x": x_train, "y": y_train})
    eval_data = NumpyDataset({"x": x_eval, "y": y_eval})
    return train_data, eval_data
