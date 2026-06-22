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

from fastestimator.op.numpyop.multivariate import RandomCrop3D


class TestRandomCrop3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crop_size = (8, 14, 14)
        cls.single_input = [np.random.rand(16, 28, 28).astype(np.float32)]
        cls.single_output_shape = (8, 14, 14)
        cls.input_image_and_mask = [
            np.random.rand(16, 28, 28).astype(np.float32), np.random.rand(16, 28, 28).astype(np.float32)
        ]
        cls.image_and_mask_output_shape = (8, 14, 14)

    def test_input(self):
        op = RandomCrop3D(inputs='x', outputs='x', crop_size=self.crop_size)
        output = op.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_input_image_and_mask(self):
        op = RandomCrop3D(inputs=('x', 'mask'), outputs=('x', 'mask'), crop_size=self.crop_size)
        output = op.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)

    def test_4d_input(self):
        data = [np.random.rand(16, 28, 28, 2).astype(np.float32)]
        op = RandomCrop3D(inputs='x', outputs='x', crop_size=self.crop_size)
        output = op.forward(data=data, state={})
        self.assertEqual(output[0].shape, (8, 14, 14, 2))

    def test_padding_when_smaller(self):
        data = [np.random.rand(4, 10, 10).astype(np.float32)]
        op = RandomCrop3D(inputs='x', outputs='x', crop_size=(16, 28, 28))
        output = op.forward(data=data, state={})
        self.assertEqual(output[0].shape, (16, 28, 28))

    def test_output_dtype(self):
        op = RandomCrop3D(inputs='x', outputs='x', crop_size=self.crop_size)
        output = op.forward(data=self.single_input, state={})
        self.assertEqual(output[0].dtype, np.float32)
