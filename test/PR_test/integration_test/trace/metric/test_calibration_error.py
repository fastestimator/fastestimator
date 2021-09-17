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

from fastestimator.test.unittest_util import is_equal, sample_system_object
from fastestimator.trace.metric import CalibrationError
from fastestimator.util import Data


class TestCalibrationError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration_error = CalibrationError(true_key='y', pred_key='y_pred')
        cls.calibration_error.system = sample_system_object()

    def test_on_epoch_begin(self):
        self.calibration_error.on_epoch_begin(data=Data())
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.calibration_error.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.calibration_error.y_pred, [])

    def test_on_batch_end(self):
        self.calibration_error.y_true = []
        self.calibration_error.y_pred = []
        batch1 = {'y': np.array([0, 0, 1, 1]), 'y_pred': np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])}
        self.calibration_error.on_batch_end(data=Data(batch1))
        with self.subTest('Check true values'):
            self.assertTrue(is_equal(self.calibration_error.y_true, list(batch1['y'])))
        with self.subTest('Check pred values'):
            self.assertTrue(is_equal(self.calibration_error.y_pred, list(batch1['y_pred'])))
        batch2 = {'y': np.array([1, 1, 0, 0]), 'y_pred': np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])}
        self.calibration_error.on_batch_end(data=Data(batch2))
        with self.subTest('Check true values (2 batches)'):
            self.assertTrue(is_equal(self.calibration_error.y_true, list(batch1['y']) + list(batch2['y'])))
        with self.subTest('Check pred values (2 batches)'):
            self.assertTrue(is_equal(self.calibration_error.y_pred, list(batch1['y_pred']) + list(batch2['y_pred'])))

    def test_on_epoch_end(self):
        self.calibration_error.y_true = [0] * 50 + [1] * 50
        self.calibration_error.y_pred = list(np.array([1.0, 0.0] * 50 + [0.0, 1.0] * 50).reshape(100, 2))
        data = Data()
        self.calibration_error.on_epoch_end(data=data)
        with self.subTest('Check if calibration error exists'):
            self.assertIn('calibration_error', data)
        with self.subTest('Check the value of calibration error'):
            self.assertEqual(0.0, data['calibration_error'])

    def test_perfect_calibration(self):
        self.calibration_error.y_true = [0] * 50 + [1] * 50
        self.calibration_error.y_pred = list(
            np.array([1.0, 0.0] * 25 + [0.5, 0.5] * 50 + [0.0, 1.0] * 25).reshape(100, 2))
        data = Data()
        self.calibration_error.on_epoch_end(data=data)
        self.assertEqual(0.0, data['calibration_error'])

    def test_imperfect_calibration(self):
        self.calibration_error.y_true = [0] * 50 + [1] * 50
        self.calibration_error.y_pred = list(np.array([1.0, 0.0] * 50 + [0.5, 0.5] * 50).reshape(100, 2))
        data = Data()
        self.calibration_error.on_epoch_end(data=data)
        self.assertEqual(0.3536, data['calibration_error'])
