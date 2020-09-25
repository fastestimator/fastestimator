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

import numpy as np

from fastestimator.op.numpyop.multivariate import RandomResizedCrop


class TestRandomResizedCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.height = 10
        cls.width = 12
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (10, 12, 3)
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (10, 12, 3)

    def test_input(self):
        resized_crop = RandomResizedCrop(image_in='x', height=self.height, width=self.width)
        output = resized_crop.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_input_image_and_mask(self):
        resized_crop = RandomResizedCrop(image_in='x', mask_in='x_mask', height=self.height, width=self.width)
        output = resized_crop.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
