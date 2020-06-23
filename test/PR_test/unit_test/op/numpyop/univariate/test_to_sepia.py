import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import ToSepia


class TestToSepia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.random.rand(28, 28, 3)]
        cls.single_output_shape = (28, 28, 3)
        cls.multi_input = [np.random.rand(28, 28, 3), np.random.rand(28, 28, 3)]
        cls.multi_output_shape = (28, 28, 3)

    def test_single_input(self):
        to_sepia = ToSepia(inputs='x', outputs='x')
        output = to_sepia.forward(data=self.single_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.single_output_shape)

    def test_multi_input(self):
        to_sepia = ToSepia(inputs='x', outputs='x')
        output = to_sepia.forward(data=self.multi_input, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output in output:
            with self.subTest('Check output mask shape'):
                self.assertEqual(img_output.shape, self.multi_output_shape)
