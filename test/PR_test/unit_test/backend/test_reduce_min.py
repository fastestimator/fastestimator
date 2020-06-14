import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import reduce_min


class TestReduceMin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, 3]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_reduce_min_np_type(self):
        self.assertIsInstance(reduce_min(self.test_np), np.ScalarType, 'Output type must be NumPy')

    def test_reduce_min_np_value(self):
        self.assertEqual(reduce_min(self.test_np), 1)

    def test_reduce_min_tf_type(self):
        self.assertIsInstance(reduce_min(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_reduce_min_tf_value(self):
        self.assertTrue(np.array_equal(reduce_min(self.test_tf).numpy(), -3))

    def test_reduce_min_torch_type(self):
        self.assertIsInstance(reduce_min(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_reduce_min_torch_value(self):
        self.assertTrue(np.array_equal(reduce_min(self.test_torch).numpy(), -3))

    def test_reduce_min_axis(self):
        self.assertTrue(np.array_equal(reduce_min(self.test_np, axis=0), [1, 2]))
