import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import CenterCrop


class TestCenterCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.height = 10
        cls.width = 10
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (10, 10, 3)
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (10, 10, 3)

    def test_single_input(self):
        cc = CenterCrop(image_in='x', height=10, width=10)
        output = cc.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_multi_input(self):
        cc = CenterCrop(height=10, width=10, image_in='x', mask_in='x_mask')
        output = cc.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
