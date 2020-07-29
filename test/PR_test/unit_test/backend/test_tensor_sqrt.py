import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import tensor_sqrt
from fastestimator.test.unittest_util import is_equal


class TestTensorSqrt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 4, 6], [4, 9, 16]])
        cls.test_output_np = np.array([[1., 2., 2.44948974], [2., 3., 4.]])
        cls.test_tf = tf.constant([[1, 4, 6], [4, 9, 16]], dtype=tf.float32)
        cls.test_output_tf = tf.constant([[1.0, 2.0, 2.4494898], [2.0, 3.0, 4.0]])
        cls.test_torch = torch.tensor([[1, 4, 6], [4, 9, 16]], dtype=torch.float32)
        cls.test_output_torch = torch.Tensor([[1., 2., 2.4495], [2., 3., 4.]])

    def test_tensor_sqrt_np_type(self):
        self.assertIsInstance(tensor_sqrt(self.test_np), np.ndarray, 'Output type must be NumPy array')

    def test_tensor_sqrt_np_value(self):
        self.assertTrue(np.allclose(tensor_sqrt(self.test_np), self.test_output_np))

    def test_tensor_sqrt_tf_type(self):
        self.assertIsInstance(tensor_sqrt(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_tensor_sqrt_tf_value(self):
        self.assertTrue(np.allclose(tensor_sqrt(self.test_tf).numpy(), self.test_output_tf.numpy()))

    def test_tensor_sqrt_torch_type(self):
        self.assertIsInstance(tensor_sqrt(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_tensor_sqrt_torch_value(self):
        self.assertTrue(torch.allclose(tensor_sqrt(self.test_torch), self.test_output_torch))
