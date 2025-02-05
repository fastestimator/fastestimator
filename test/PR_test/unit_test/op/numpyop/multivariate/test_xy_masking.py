# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.op.numpyop.multivariate import XYMasking

class TestXYMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (28, 28, 3)
        cls.input_image_and_mask_ones = [np.ones((28, 28, 3)), np.ones((28, 28, 3))]

    def test_input_image_and_mask(self):
        xy_masking = XYMasking(image_in='x', mask_in='x_mask', num_masks_x=1, num_masks_y=1, mask_x_length=5, mask_y_length=5)
        output = xy_masking.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)

    def test_max_objects_image_and_mask(self):
        xy_masking = XYMasking(image_in='x', mask_in='x_mask', num_masks_x=2, num_masks_y=2, mask_x_length=5, mask_y_length=5)
        output = xy_masking.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)

    def test_max_objects_masking_values(self):
        num_masks = 1
        xy_masking = XYMasking(image_in='x', mask_in='x_mask', num_masks_x=num_masks, num_masks_y=num_masks, mask_x_length=1, mask_y_length=1)
        output = xy_masking.forward(data=self.input_image_and_mask_ones, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask values'):
            mask_val_count = np.prod(self.image_and_mask_output_shape) * num_masks
            actual_non_zero_count = np.count_nonzero(output[1])
            self.assertEqual(actual_non_zero_count, mask_val_count)
