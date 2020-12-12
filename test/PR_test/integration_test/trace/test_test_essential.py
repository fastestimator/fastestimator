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

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace import TestEssential
from fastestimator.util.data import Data


class TestTestEssential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})

    def test_on_epoch_begin(self):
        test_essential = TestEssential(monitor_names='loss')
        test_essential.system = sample_system_object()
        test_essential.on_epoch_begin(data=self.data)
        self.assertIsNone(test_essential.test_results)

    def test_on_batch_end_test_results_not_none(self):
        test_essential = TestEssential(monitor_names='loss')
        test_essential.system = sample_system_object()
        test_essential.test_results = {'loss': [95]}
        test_essential.on_batch_end(data=self.data)
        self.assertEqual(test_essential.test_results['loss'], [95, 10])

    def test_on_batch_end_test_results_none(self):
        data = Data({'loss': 5})
        test_essential = TestEssential(monitor_names='loss')
        test_essential.system = sample_system_object()
        test_essential.on_batch_end(data=data)
        self.assertEqual(test_essential.test_results['loss'], [5])

    def test_on_epoch_end(self):
        data = Data({})
        test_essential = TestEssential(monitor_names='loss')
        test_essential.system = sample_system_object()
        test_essential.test_results = {'loss': [10, 20]}
        test_essential.on_epoch_end(data=data)
        self.assertEqual(data['loss'], 15.0)
