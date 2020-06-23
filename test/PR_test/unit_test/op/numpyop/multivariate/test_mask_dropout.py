import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import MaskDropout


class TestMaskDropout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (28, 28, 3)

    def test_input_image_and_mask(self):
        mask_dropout = MaskDropout(image_in='x', mask_in='x_mask')
        output = mask_dropout.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)

    def test_max_objects_image_and_mask(self):
        mask_dropout = MaskDropout(image_in='x', mask_in='x_mask', max_objects=(2, 5))
        output = mask_dropout.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
