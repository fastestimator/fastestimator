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

from fastestimator.trace.metric import Accuracy
from fastestimator.util import Data


class TestAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        x_1d = np.array([2.5])
        x_pred_1d = np.array([1])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.data_1d = Data({'x': x_1d, 'x_pred': x_pred_1d})
        cls.accuracy = Accuracy(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.accuracy.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of correct'):
            self.assertEqual(self.accuracy.correct, 0)
        with self.subTest('Check initial value of total'):
            self.assertEqual(self.accuracy.total, 0)

    def test_on_batch_end(self):
        self.accuracy.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.accuracy.correct, 1)
        with self.subTest('Check total values'):
            self.assertEqual(self.accuracy.total, 3)

    def test_on_epoch_end(self):
        self.accuracy.correct = 1
        self.accuracy.total = 3
        self.accuracy.on_epoch_end(data=self.data)
        with self.subTest('Check if accuracy value exists'):
            self.assertIn('accuracy', self.data)
        with self.subTest('Check the value of accuracy'):
            self.assertEqual(round(self.data['accuracy'], 2), 0.33)

    def test_1d_data_on_batch_end(self):
        self.accuracy.on_batch_end(data=self.data_1d)
        with self.subTest('Check correct values'):
            self.assertEqual(self.accuracy.correct, 0)
        with self.subTest('Check total values'):
            self.assertEqual(self.accuracy.total, 1)

    def test_1d_data_on_epoch_end(self):
        self.accuracy.correct = 0
        self.accuracy.total = 1
        self.accuracy.on_epoch_end(data=self.data_1d)
        with self.subTest('Check if accuracy value exists'):
            self.assertIn('accuracy', self.data_1d)
        with self.subTest('Check the value of accuracy'):
            self.assertEqual(round(self.data_1d['accuracy'], 2), 0.0)
