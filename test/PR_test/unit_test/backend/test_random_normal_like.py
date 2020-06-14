import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import random_normal_like


class TestRandomNormalLike(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[0, 1], [1, 1]])
        cls.test_tf = tf.constant([[0, 1], [2, 2]])
        cls.test_torch = torch.Tensor([[1, 1], [2, 3]])

    def test_random_normal_np_type(self):
        self.assertIsInstance(random_normal_like(self.test_np), np.ndarray, 'Output must be NumPy Array')

    def test_random_normal_np_value(self):
        self.assertTrue((random_normal_like(self.test_np).shape == (2, 2)),
                        'Output array shape should be same as input')

    def test_random_normal_tf_type(self):
        self.assertIsInstance(random_normal_like(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_random_normal_tf_value(self):
        output_shape = tf.shape([2, 2])
        self.assertTrue(tf.reduce_all(tf.equal(tf.shape(random_normal_like(self.test_tf)), output_shape)),
                        'Output tensor shape should be same as input')

    def test_random_normal_torch_type(self):
        self.assertIsInstance(random_normal_like(self.test_torch), torch.Tensor, 'Output must be torch.Tensor')

    def test_random_normal_torch_value(self):
        output_shape = (2, 2)
        self.assertTrue((random_normal_like(self.test_torch).size() == output_shape),
                        'Output tensor shape should be same as input')
