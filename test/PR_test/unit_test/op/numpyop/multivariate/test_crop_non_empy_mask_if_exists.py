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

from fastestimator.op.numpyop.multivariate import CropNonEmptyMaskIfExists


class TestCenterCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.height = 10
        cls.width = 10
        cls.input = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.output_shape = (10, 10, 3)

    def test_input(self):
        crop_mask = CropNonEmptyMaskIfExists(height=self.height, width=self.width, image_in='x', mask_in='x_mask')
        output = crop_mask.forward(data=self.input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.output_shape)
