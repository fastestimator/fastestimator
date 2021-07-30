# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.op.numpyop.univariate import ShearY


class TestShearY(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.randint(0, 256, size=(28, 28, 3)).astype(np.uint8)]
        cls.single_output_shape = (28, 28, 3)
        cls.multi_input = [
            np.random.randint(0, 256, size=(28, 28, 3)).astype(np.uint8),
            np.random.randint(0, 256, size=(28, 28, 3)).astype(np.uint8)
        ]
        cls.multi_output_shape = (28, 28, 3)

    def test_single_input(self):
        shear_y = ShearY(inputs='x', outputs='x')
        output = shear_y.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_multi_input(self):
        shear_y = ShearY(inputs='x', outputs='x')
        output = shear_y.forward(data=self.multi_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output in output:
            with self.subTest('Check output image shape'):
                self.assertEqual(img_output.shape, self.multi_output_shape)
