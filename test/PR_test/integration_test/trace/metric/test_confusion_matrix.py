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
from fastestimator.trace.metric import ConfusionMatrix
from fastestimator.util import Data


class TestConfusionMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 2], [3, 4]])
        x_pred = np.array([[1, 5, 3], [2, 1, 0]])
        cls.data = Data({'x': x, 'x_pred': x_pred})
        cls.matrix = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        cls.confusion_matrix = ConfusionMatrix(true_key='x', pred_key='x_pred', num_classes=3)

    def test_on_epoch_begin(self):
        self.confusion_matrix.on_epoch_begin(data=self.data)
        self.assertEqual(self.confusion_matrix.matrix, None)

    def test_on_batch_end(self):
        self.confusion_matrix.on_batch_end(data=self.data)
        self.assertTrue(is_equal(self.confusion_matrix.matrix, self.matrix))

    def test_on_epoch_end(self):
        self.confusion_matrix.matrix = self.matrix
        self.confusion_matrix.on_epoch_end(data=self.data)
        with self.subTest('Check if confusion matrix value exists'):
            self.assertIn('confusion_matrix', self.data)
        with self.subTest('Check the value of matrix'):
            self.assertTrue(is_equal(self.data['confusion_matrix'], self.matrix))

    def test_on_batch_end_matrix_not_none(self):
        self.confusion_matrix.matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        matrix_output = np.array([[0, 0, 0], [1, 2, 0], [0, 0, 0]])
        self.confusion_matrix.on_batch_end(data=self.data)
        self.assertTrue(is_equal(self.confusion_matrix.matrix, matrix_output))
