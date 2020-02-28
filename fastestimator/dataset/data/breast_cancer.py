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
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fastestimator.dataset.numpy_dataset import NumpyDataset


def load_data() -> Tuple[NumpyDataset, NumpyDataset]:
    (X, y) = load_breast_cancer(True)
    x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)

    train_data = {"x": x_train, "y": np.expand_dims(y_train, -1)}
    eval_data = {"x": x_eval, "y": np.expand_dims(y_eval, -1)}

    return train_data, eval_data
