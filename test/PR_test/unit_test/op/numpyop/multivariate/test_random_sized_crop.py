import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import RandomSizedCrop


class TestRandomSizedCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.min_max_height = (20, 25)
        cls.height = 20
        cls.width = 20
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (20, 20, 3)
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (20, 20, 3)

    def test_input(self):
        crop = RandomSizedCrop(image_in='x', height=self.height, width=self.width, min_max_height=self.min_max_height)
        output = crop.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_input_image_and_mask(self):
        crop = RandomSizedCrop(image_in='x',
                               mask_in='x_mask',
                               height=self.height,
                               width=self.width,
                               min_max_height=self.min_max_height)
        output = crop.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.image_and_mask_output_shape)
        with self.subTest('Check output mask shape'):
            self.assertEqual(output[1].shape, self.image_and_mask_output_shape)
