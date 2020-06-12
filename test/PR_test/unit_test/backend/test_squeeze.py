import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import squeeze


class TestSqueeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[[[1], [2]]], [[[3], [3]]]])
        cls.test_output_np = np.array([[1, 2], [3, 3]])
        cls.test_output_np_axis = np.array([[[1], [2]], [[3], [3]]])
        cls.test_tf = tf.constant([[[1], [1]], [[2], [-3]], [[4], [1]]])
        cls.test_output_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[[1], [1]], [[2], [-3]], [[4], [1]]])
        cls.test_output_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_squeeze_np_type(self):
        self.assertIsInstance(squeeze(self.test_np), np.ndarray, 'Output type must be NumPy array')

    def test_squeeze_np_value(self):
        self.assertTrue(np.array_equal(squeeze(self.test_np), self.test_output_np))

    def test_squeeze_tf_type(self):
        self.assertIsInstance(squeeze(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_squeeze_tf_value(self):
        self.assertTrue(np.array_equal(squeeze(self.test_tf).numpy(), self.test_output_tf))

    def test_squeeze_torch_type(self):
        self.assertIsInstance(squeeze(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_squeeze_torch_value(self):
        self.assertTrue(np.array_equal(squeeze(self.test_torch).numpy(), self.test_output_torch))

    def test_squeeze_axis(self):
        self.assertTrue(np.array_equal(squeeze(self.test_np, axis=1), self.test_output_np_axis))
