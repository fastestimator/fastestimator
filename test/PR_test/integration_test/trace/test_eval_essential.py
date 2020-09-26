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
from fastestimator.trace import EvalEssential
from fastestimator.util.data import Data


class TestEvalEssential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})

    def test_on_epoch_begin(self):
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.on_epoch_begin(data=self.data)
        self.assertIsNone(eval_essential.eval_results)

    def test_on_batch_end_eval_results_not_none(self):
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.eval_results = {'loss': [95]}
        eval_essential.on_batch_end(data=self.data)
        self.assertEqual(eval_essential.eval_results['loss'], [95, 10])

    def test_on_batch_end_eval_results_none(self):
        data = Data({'loss': 5})
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.on_batch_end(data=data)
        self.assertEqual(eval_essential.eval_results['loss'], [5])

    def test_on_epoch_end(self):
        data = Data({})
        eval_essential = EvalEssential(monitor_names='loss')
        eval_essential.system = sample_system_object()
        eval_essential.eval_results = {'loss': [10, 20]}
        eval_essential.on_epoch_end(data=data)
        self.assertEqual(data['loss'], 15.0)
