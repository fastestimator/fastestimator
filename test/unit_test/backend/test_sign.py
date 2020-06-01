import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import sign


class TestSign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, -3]])
        cls.test_output_np = np.array([[1, 1], [1, -1]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_output_tf = tf.constant([[1, 1], [1, -1], [1, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])
        cls.test_output_torch = torch.Tensor([[1, 1], [1, -1], [1, 1]])

    def test_sign_np(self):
        output = sign(self.test_np)
        # check the output type
        self.assertIsInstance(output, np.ndarray, 'Output type must be NumPy array')
        # check the output value
        self.assertTrue(np.array_equal(output, self.test_output_np))

    def test_sign_tf(self):
        output = sign(self.test_tf)
        # check the output type
        self.assertIsInstance(output, tf.Tensor, 'Output type must be tf.Tensor')
        # check the output value
        self.assertTrue(np.array_equal(output.numpy(), self.test_output_tf))

    def test_sign_torch(self):
        output = sign(self.test_torch)
        # check the output type
        self.assertIsInstance(output, torch.Tensor, 'Output type must be torch.Tensor')
        # check the output value
        self.assertTrue(np.array_equal(output.numpy(), self.test_output_torch))


if __name__ == "__main__":
    unittest.main()