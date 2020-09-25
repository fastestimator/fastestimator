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
from io import StringIO
from unittest.mock import patch

import numpy as np

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.adapt import EarlyStopping
from fastestimator.util.data import Data


class TestEarlyStopping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})
        cls.expected_msg = "FastEstimator-EarlyStopping: 'loss' triggered an early stop. Its best value was 2 at epoch 2"

    def test_on_begin_compare_min(self):
        early_stopping = EarlyStopping()
        early_stopping.system = sample_system_object()
        early_stopping.on_begin(data=self.data)
        self.assertEqual(early_stopping.best, np.Inf)

    def test_on_begin_compare_max(self):
        early_stopping = EarlyStopping(compare='max')
        early_stopping.system = sample_system_object()
        early_stopping.on_begin(data=self.data)
        self.assertEqual(early_stopping.best, -np.Inf)

    def test_on_begin_baseline_arbitrary_value(self):
        early_stopping = EarlyStopping(baseline=5.0)
        early_stopping.system = sample_system_object()
        early_stopping.on_begin(data=self.data)
        self.assertEqual(early_stopping.best, 5.0)

    def test_on_epoch_end_early_stopping_msg(self):
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            early_stopping = EarlyStopping(baseline=5.0)
            early_stopping.system = sample_system_object()
            early_stopping.system.epoch_idx = 3
            early_stopping.best = 2
            early_stopping.on_epoch_end(data=self.data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, self.expected_msg)

    def test_on_epoch_end_monitor_op(self):
        early_stopping = EarlyStopping(baseline=5.0)
        early_stopping.system = sample_system_object()
        early_stopping.min_delta = 1
        early_stopping.monitor_op = np.greater
        early_stopping.best = 7
        early_stopping.on_epoch_end(data=self.data)
        with self.subTest('Check value of wait'):
            self.assertEqual(early_stopping.wait, 0)
        with self.subTest('Check value of best'):
            self.assertEqual(early_stopping.best, 10)
