import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import reshape


class TestReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, 3]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_reshape_np(self):
        output = reshape(self.test_np, shape=(4, ))
        # check the output type
        self.assertIsInstance(output, (np.ndarray, np.ScalarType), 'Output type must be NumPy')
        # check the output value
        self.assertEqual(output.shape, (4, ))

    def test_reshape_tf(self):
        output = reshape(self.test_tf, shape=(2, 3))
        # check the output type
        self.assertIsInstance(output, tf.Tensor, 'Output type must be tf.Tensor')
        # check the output value
        self.assertEqual(output.numpy().shape, (2, 3))

    def test_reshape_torch(self):
        output = reshape(self.test_torch, shape=(2, 3))
        # check the output type
        self.assertIsInstance(output, torch.Tensor, 'Output type must be torch.Tensor')
        # check the output value
        self.assertEqual(output.numpy().shape, (2, 3))


if __name__ == "__main__":
    unittest.main()