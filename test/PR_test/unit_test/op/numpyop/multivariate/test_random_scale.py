import unittest

import numpy as np

from fastestimator.op.numpyop.multivariate import RandomScale


class TestRandomScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (28, 28, 3)
        cls.input_image_and_mask = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.image_and_mask_output_shape = (27, 27, 3)

    def test_input(self):
        random_scale = RandomScale(image_in='x')
        output = random_scale.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)

    def test_input_image_and_mask(self):
        random_scale = RandomScale(image_in='x', mask_in='x_mask')
        output = random_scale.forward(data=self.input_image_and_mask, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
