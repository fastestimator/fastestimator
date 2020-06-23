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
