import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import SmallestMaxSize


class TestSmallesMaxSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.max_size = 512
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (512, 512, 3)
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (512, 512, 3)

    def test_input(self):
        maxsize = SmallestMaxSize(image_in='x', max_size=self.max_size)
        output = maxsize.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_input_image_and_mask(self):
        maxsize = SmallestMaxSize(image_in='x', mask_in='x_mask', max_size=self.max_size)
        output = maxsize.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
