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

from fastestimator.op.numpyop.univariate import Calibrate
from fastestimator.test.unittest_util import is_equal


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.array([1, 2, 3, 5])]
        cls.single_output = [np.array([0.5, 1.0, 1.5, 2.5])]
        cls.multi_input = [np.array([2, 2]), np.array([0, 1, 2])]
        cls.multi_output = [np.array([1, 1]), np.array([0, 0.5, 1])]

    def test_single_input(self):
        op = Calibrate(inputs='x', outputs='x', calibration_fn=lambda x: x/2)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = Calibrate(inputs=['x', 'y'], outputs=['x', 'y'], calibration_fn=lambda x: x/2)
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))

    def test_single_input_fn_from_disk(self):
        tmpdirname = tempfile.mkdtemp()
        fn_path = os.path.join(tmpdirname, 'fn.pkl')
        fn = lambda x: x/2
        with open(fn_path, 'wb') as f:
            dill.dump(fn, f)
        op = Calibrate(inputs='x', outputs='x', calibration_fn=fn_path)
        with self.subTest("Do nothing on Warmup"):
            resp = op.forward(self.single_input, state={'warmup': True})
            self.assertTrue(np.array_equal(resp, self.single_input))
        with self.subTest("Load function during regular execution"):
            resp = op.forward(self.single_input, state={'warmup': False})
            self.assertTrue(np.array_equal(resp, self.single_output))
        os.remove(fn_path)
        with self.subTest("Continue to use function without re-loading"):
            resp = op.forward(self.single_input, state={'warmup': False})
            self.assertTrue(np.array_equal(resp, self.single_output))
