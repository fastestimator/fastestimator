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

from fastestimator.op.numpyop.univariate import CTWindow


class TestCTWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.uniform(-1000, 3000, (28, 28)).astype(np.float32)]
        cls.single_output_shape = (28, 28)
        cls.multi_input = [
            np.random.uniform(-1000, 3000, (28, 28)).astype(np.float32),
            np.random.uniform(-1000, 3000, (28, 28)).astype(np.float32)
        ]
        cls.multi_output_shape = (28, 28)

    def test_single_input(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        output = op.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_multi_input(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        output = op.forward(data=self.multi_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output in output:
            with self.subTest('Check output image shape'):
                self.assertEqual(img_output.shape, self.multi_output_shape)

    def test_output_range(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        output = op.forward(data=self.single_input, state={})
        with self.subTest('Check output min'):
            self.assertGreaterEqual(output[0].min(), 0.0)
        with self.subTest('Check output max'):
            self.assertLessEqual(output[0].max(), 1.0)

    def test_custom_output_range(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50, output_min=-1.0, output_max=1.0)
        output = op.forward(data=self.single_input, state={})
        with self.subTest('Check output min'):
            self.assertGreaterEqual(output[0].min(), -1.0)
        with self.subTest('Check output max'):
            self.assertLessEqual(output[0].max(), 1.0)

    def test_output_dtype(self):
        op = CTWindow(inputs='x', outputs='x', window_width=400, window_level=50)
        output = op.forward(data=self.single_input, state={})
        self.assertEqual(output[0].dtype, np.float32)
