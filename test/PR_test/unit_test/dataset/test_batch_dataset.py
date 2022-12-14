# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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
import unittest

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.dataset import GeneratorDataset


def inputs():
    while True:
        yield {'x': np.random.rand(16), 'y': np.random.randint(16)}


class TestBatchDataset(unittest.TestCase):
    def test_dataset(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        unpaired_ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=[2, 2])

        self.assertEqual(len(unpaired_ds), 5)

    def test_dataset_probability(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        ds2 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        unpaired_ds = fe.dataset.BatchDataset(datasets=[ds1, ds2], num_samples=4, probability=[0.5, 0.5])

        self.assertEqual(len(unpaired_ds), 5)

    def test_split(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})
        train_data.split(0.1)

        self.assertEqual(len(train_data), 54000)
