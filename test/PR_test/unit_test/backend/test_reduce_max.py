import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import reduce_max


class TestReduceMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, 3]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_reduce_max_np(self):
        output = reduce_max(self.test_np)
        # check the output type
        self.assertIsInstance(output, np.ScalarType, 'Output type must be NumPy')
        # check the output value
        self.assertEqual(output, 3)

    def test_reduce_max_tf(self):
        output = reduce_max(self.test_tf)
        # check the output type
        self.assertIsInstance(output, tf.Tensor, 'Output type must be tf.Tensor')
        # check the output value
        self.assertTrue(np.array_equal(output.numpy(), 4))

    def test_reduce_max_torch(self):
        output = reduce_max(self.test_torch)
        # check the output type
        self.assertIsInstance(output, torch.Tensor, 'Output type must be torch.Tensor')
        # check the output value
        self.assertTrue(np.array_equal(output.numpy(), 4))

    def test_reduce_max_axis(self):
        output = reduce_max(self.test_np, axis=0)
        # check the output type
        self.assertIsInstance(output, np.ndarray, 'Output type should be NumPy array')
        # check output value
        self.assertTrue(np.array_equal(output, [3, 3]))
