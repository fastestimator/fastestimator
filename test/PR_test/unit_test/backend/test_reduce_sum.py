import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import reduce_sum


class TestReduceSum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, 3]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_reduce_sum_np_type(self):
        self.assertIsInstance(reduce_sum(self.test_np), np.ScalarType, 'Output type must be NumPy')

    def test_reduce_sum_np_value(self):
        self.assertEqual(reduce_sum(self.test_np), 9)

    def test_reduce_sum_tf_type(self):
        self.assertIsInstance(reduce_sum(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_reduce_sum_tf_value(self):
        self.assertTrue(np.array_equal(reduce_sum(self.test_tf).numpy(), 6))

    def test_reduce_sum_torch_type(self):
        self.assertIsInstance(reduce_sum(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_reduce_sum_torch_value(self):
        self.assertTrue(np.array_equal(reduce_sum(self.test_torch).numpy(), 6))

    def test_reduce_sum_axis(self):
        self.assertTrue(np.array_equal(reduce_sum(self.test_np, axis=0), [4, 5]))
