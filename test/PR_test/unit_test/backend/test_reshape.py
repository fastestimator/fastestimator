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

    def test_reshape_np_type(self):
        self.assertIsInstance(reshape(self.test_np, shape=(4, )), (np.ndarray, np.ScalarType),
                              'Output type must be NumPy')

    def test_reshape_np_value(self):
        self.assertEqual(reshape(self.test_np, shape=(4, )).shape, (4, ))

    def test_reshape_tf_type(self):
        self.assertIsInstance(reshape(self.test_tf, shape=(2, 3)), tf.Tensor, 'Output type must be tf.Tensor')

    def test_reshape_tf_value(self):
        self.assertEqual(reshape(self.test_tf, shape=(2, 3)).numpy().shape, (2, 3))

    def test_reshape_torch_type(self):
        self.assertIsInstance(reshape(self.test_torch, shape=(2, 3)), torch.Tensor, 'Output type must be torch.Tensor')

    def test_reshape_torch_value(self):
        self.assertEqual(reshape(self.test_torch, shape=(2, 3)).numpy().shape, (2, 3))
