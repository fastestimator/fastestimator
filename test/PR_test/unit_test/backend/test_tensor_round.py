import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import tensor_round
from fastestimator.test.unittest_util import is_equal


class TestTensorRound(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1.25, 4.5, 6], [4, 9.11, 16]])
        cls.test_output_np = np.array([[1, 4, 6], [4, 9, 16]])
        cls.test_tf = tf.constant([[1.25, 4.5, 6], [4, 9.11, 16.9]])
        cls.test_output_tf = tf.constant([[1, 4, 6], [4, 9, 17]])
        cls.test_torch = torch.Tensor([[1.25, 4.5, 6], [4, 9.11, 16]])
        cls.test_output_torch = torch.Tensor([[1, 4, 6], [4, 9, 16]])

    def test_tensor_round_np_type(self):
        self.assertIsInstance(tensor_round(self.test_np), np.ndarray, 'Output type must be NumPy array')

    def test_tensor_round_np_value(self):
        self.assertTrue(is_equal(tensor_round(self.test_np), self.test_output_np))

    def test_tensor_round_tf_type(self):
        self.assertIsInstance(tensor_round(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_tensor_round_tf_value(self):
        self.assertTrue(is_equal(tensor_round(self.test_tf), self.test_output_tf))

    def test_tensor_round_torch_type(self):
        self.assertIsInstance(tensor_round(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_sign_torch_value(self):
        self.assertTrue(is_equal(tensor_round(self.test_torch), self.test_output_torch))
