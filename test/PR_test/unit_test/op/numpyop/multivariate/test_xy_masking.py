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
            
    def test_fill_values(self):
        input_image = np.ones((28, 28, 3), dtype=np.float32)
        input_mask = np.ones((28, 28, 3), dtype=np.float32)
        fill_value = 0.5
        mask_fill_value = 0.2
        xy_masking = XYMasking(image_in='x', mask_in='x_mask', num_masks_x=1, num_masks_y=1, mask_x_length=5, mask_y_length=5, fill_value=fill_value, mask_fill_value=mask_fill_value)
        
        output_image, output_mask = xy_masking.forward(data=[input_image, input_mask], state={})

        with self.subTest('Check output type'):
            self.assertEqual(type(output_image), np.ndarray)
            self.assertEqual(type(output_mask), np.ndarray)
        with self.subTest('Check output image shape'):
            self.assertEqual(output_image.shape, input_image.shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output_mask.shape, input_mask.shape)
            
        masked_indices = np.where(output_image == fill_value)
        with self.subTest('Check fill value in image'):
            self.assertTrue(np.any(output_image == fill_value), f"Fill value {fill_value} not found in image")
        with self.subTest('Check fill value in mask'):
            self.assertTrue(np.any(output_mask == mask_fill_value), f"Mask fill value {mask_fill_value} not found in mask")
        with self.subTest('Check that the masked regions in image and mask correspond'):
            self.assertTrue(np.array_equal(masked_indices, np.where(output_mask == mask_fill_value)), "Masked regions in image and mask do not correspond")