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

from fastestimator.op.numpyop.univariate import RandomShapes


class TestRandomShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.rand(28, 28, 3).astype(np.float32)]
        cls.single_output_shape = (28, 28, 3)
        cls.multi_input = [np.random.rand(28, 28, 3).astype(np.float32), np.random.rand(28, 28, 3).astype(np.float32)]
        cls.multi_output_shape = (28, 28, 3)
        cls.single_input_bw = [np.random.rand(32, 40, 1).astype(np.float32)]
        cls.single_bw_output_shape = (32, 40, 1)
        cls.single_input_int = [np.random.randint(0, 255, (17, 33, 1))]
        cls.single_int_shape = (17, 33, 1)

    def test_single_input(self):
        random_shapes = RandomShapes(inputs='x', outputs='x')
        output = random_shapes.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)
        with self.subTest('Check output type'):
            self.assertEqual(output[0].dtype, self.single_input[0].dtype)

    def test_multi_input(self):
        random_shapes = RandomShapes(inputs='x', outputs='x')
        output = random_shapes.forward(data=self.multi_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output, image_input in zip(output, self.multi_input):
            with self.subTest('Check output shape'):
                self.assertEqual(img_output.shape, self.multi_output_shape)
            with self.subTest('Check output type'):
                self.assertEqual(img_output.dtype, image_input.dtype)

    def test_single_input_black_white(self):
        random_shapes = RandomShapes(inputs='x', outputs='x')
        output = random_shapes.forward(data=self.single_input_bw, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_bw_output_shape)
        with self.subTest('Check output type'):
            self.assertEqual(output[0].dtype, self.single_input_bw[0].dtype)

    def test_single_input_int(self):
        random_shapes = RandomShapes(inputs='x', outputs='x')
        output = random_shapes.forward(data=self.single_input_int, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_int_shape)
        with self.subTest('Check output type'):
            self.assertEqual(output[0].dtype, self.single_input_int[0].dtype)
