# Copyright 2026 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.op.numpyop.univariate.ct_window import CTWindow
from fastestimator.test.unittest_util import is_equal


class TestCTWindowIntegration(unittest.TestCase):
    def test_soft_tissue_window(self):
        """Soft tissue window: W=400, L=50. Range = [-150, 250]."""
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        data = [np.array([-150.0, 50.0, 250.0], dtype=np.float32)]
        output = op.forward(data=data, state={})
        expected = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        self.assertTrue(is_equal(output[0], expected))

    def test_clipping_below(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        data = [np.array([-1000.0], dtype=np.float32)]
        output = op.forward(data=data, state={})
        self.assertAlmostEqual(float(output[0][0]), 0.0)

    def test_clipping_above(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        data = [np.array([3000.0], dtype=np.float32)]
        output = op.forward(data=data, state={})
        self.assertAlmostEqual(float(output[0][0]), 1.0)

    def test_multi_input_consistency(self):
        """Multiple inputs should all be windowed identically (deterministic)."""
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        arr = np.array([0.0, 100.0, -100.0], dtype=np.float32)
        data = [arr.copy(), arr.copy()]
        output = op.forward(data=data, state={})
        self.assertTrue(is_equal(output[0], output[1]))
