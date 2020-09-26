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

from fastestimator.test.unittest_util import is_equal
from fastestimator.trace.metric import F1Score
from fastestimator.util import Data


class TestF1Score(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        x_binary = np.array([1])
        x_pred_binary = np.array([0.9])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.data_binary = Data({'x': x_binary, 'x_pred': x_pred_binary})
        cls.f1score = F1Score(true_key='x', pred_key='x_pred')
        cls.f1score_output = np.array([0., 0.67])

    def test_on_epoch_begin(self):
        self.f1score.on_epoch_begin(data=self.data)
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.f1score.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.f1score.y_pred, [])

    def test_on_batch_end(self):
        self.f1score.y_true = []
        self.f1score.y_pred = []
        self.f1score.on_batch_end(data=self.data)
        with self.subTest('Check correct values'):
            self.assertEqual(self.f1score.y_true, [1, 1])
        with self.subTest('Check total values'):
            self.assertEqual(self.f1score.y_pred, [1, 0])

    def test_on_epoch_end(self):
        self.f1score.y_true = [1, 1]
        self.f1score.y_pred = [1, 0]
        self.f1score.binary_classification = False
        self.f1score.on_epoch_end(data=self.data)
        with self.subTest('Check if f1_score exists'):
            self.assertIn('f1_score', self.data)
        with self.subTest('Check the value of f1 score'):
            self.assertTrue(is_equal(np.round(self.data['f1_score'], 2), self.f1score_output))

    def test_on_batch_end_binary_classification(self):
        self.f1score.y_true = []
        self.f1score.y_pred = []
        self.f1score.on_batch_end(data=self.data_binary)
        with self.subTest('Check correct values'):
            self.assertEqual(self.f1score.y_true, [1])
        with self.subTest('Check total values'):
            self.assertEqual(self.f1score.y_pred, [1.0])

    def test_on_epoch_end_binary_classification(self):
        self.f1score.y_true = [1]
        self.f1score.y_pred = [1.0]
        self.f1score.binary_classification = True
        self.f1score.on_epoch_end(data=self.data_binary)
        with self.subTest('Check if f1_score exists'):
            self.assertIn('f1_score', self.data_binary)
        with self.subTest('Check the value of f1 score'):
            self.assertEqual(np.round(self.data_binary['f1_score'], 2), 1.0)
