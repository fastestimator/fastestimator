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
import os
import tempfile
import unittest

import dill
import numpy as np

from fastestimator.test.unittest_util import is_equal
from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.adapt import PBMCalibrator
from fastestimator.util.data import Data


class TestPBMCalibrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        save_dir = tempfile.mkdtemp()
        cls.save_file = os.path.join(save_dir, 'calibrator.pkl')
        cls.pbm_calibrator = PBMCalibrator(true_key='y', pred_key='y_pred', save_path=cls.save_file)
        cls.pbm_calibrator.system = sample_system_object()

    def test_on_epoch_begin(self):
        self.pbm_calibrator.on_epoch_begin(data=Data())
        with self.subTest('Check initial value of y_true'):
            self.assertEqual(self.pbm_calibrator.y_true, [])
        with self.subTest('Check initial value of y_pred'):
            self.assertEqual(self.pbm_calibrator.y_pred, [])

    def test_on_batch_end(self):
        self.pbm_calibrator.y_true = []
        self.pbm_calibrator.y_pred = []
        batch1 = {'y': np.array([0, 0, 1, 1]), 'y_pred': np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])}
        self.pbm_calibrator.on_batch_end(data=Data(batch1))
        with self.subTest('Check true values'):
            self.assertTrue(is_equal(self.pbm_calibrator.y_true, list(batch1['y'])))
        with self.subTest('Check pred values'):
            self.assertTrue(is_equal(self.pbm_calibrator.y_pred, list(batch1['y_pred'])))
        batch2 = {'y': np.array([1, 1, 0, 0]), 'y_pred': np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])}
        self.pbm_calibrator.on_batch_end(data=Data(batch2))
        with self.subTest('Check true values (2 batches)'):
            self.assertTrue(is_equal(self.pbm_calibrator.y_true, list(batch1['y']) + list(batch2['y'])))
        with self.subTest('Check pred values (2 batches)'):
            self.assertTrue(is_equal(self.pbm_calibrator.y_pred, list(batch1['y_pred']) + list(batch2['y_pred'])))

    def test_on_epoch_end(self):
        self.pbm_calibrator.y_true = [0] * 50 + [1] * 50
        self.pbm_calibrator.y_pred = list(np.array([1.0, 0.0] * 50 + [0.0, 1.0] * 50).reshape(100, 2))
        expected = np.array([1.0, 0.0] * 50 + [0.0, 1.0] * 50).reshape(100, 2)
        data = Data()
        self.pbm_calibrator.on_epoch_end(data=data)
        with self.subTest('Check if output exists'):
            self.assertIn('y_pred_calibrated', data)
        with self.subTest('Check save file exists'):
            self.assertTrue(os.path.exists(self.save_file))
        with self.subTest('Check the calibrated values'):
            self.assertTrue(np.allclose(data['y_pred_calibrated'], expected))
        with self.subTest('Check the save file performance'):
            with open(self.save_file, 'rb') as f:
                fn = dill.load(f)
                resp = fn(expected)
                self.assertTrue(np.array_equal(resp, data['y_pred_calibrated']))
