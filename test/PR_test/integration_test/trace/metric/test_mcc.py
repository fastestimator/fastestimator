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

from fastestimator.trace.metric import MCC
from fastestimator.util import Data


class TestMCC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        x_1d = np.array([2.5])
        x_pred_1d = np.array([1])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.data_1d = Data({'x': x_1d, 'x_pred': x_pred_1d})
        cls.mcc = MCC(true_key='x', pred_key='x_pred')

    def test_on_epoch_begin(self):
        self.mcc.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.mcc.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.mcc.y_pred, [])

    def test_on_batch_end(self):
        self.mcc.y_true = []
        self.mcc.y_pred = []
        self.mcc.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.mcc.y_true, [1, 1])
        with self.subTest('Check total values'):
            self.assertEqual(self.mcc.y_pred, [1, 0])

    def test_on_epoch_end(self):
        self.mcc.y_true = [2, 1]
        self.mcc.y_pred = [1, 0]
        self.mcc.on_epoch_end(data=self.data)
        with self.subTest('Check if mcc exists'):
            self.assertIn('mcc', self.data)
        with self.subTest('Check the value of mcc'):
            self.assertEqual(self.data['mcc'], -0.5)

    def test_1d_data_on_batch_end(self):
        self.mcc.y_true = []
        self.mcc.y_pred = []
        self.mcc.on_batch_end(data=self.data_1d)
        with self.subTest('Check correct values'):
            self.assertEqual(self.mcc.y_true, [2.5])
        with self.subTest('Check total values'):
            self.assertEqual(self.mcc.y_pred, [1])
