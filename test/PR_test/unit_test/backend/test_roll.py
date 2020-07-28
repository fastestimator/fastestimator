import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import roll
from fastestimator.test.unittest_util import is_equal


class TestRoll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        cls.test_output_np_1d = np.array([[5, 6, 7], [1, 2, 3]])
        cls.test_output_np_2d = np.array([[6, 7, 5], [2, 3, 1]])
        cls.test_tf = tf.constant([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        cls.test_output_tf_1d = tf.constant([[5, 6, 7], [1, 2, 3]])
        cls.test_output_tf_2d = tf.constant([[6, 7, 5], [2, 3, 1]])
        cls.test_torch = torch.Tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        cls.test_output_torch_1d = torch.Tensor([[5, 6, 7], [1, 2, 3]])
        cls.test_output_torch_2d = torch.Tensor([[6, 7, 5], [2, 3, 1]])

    def test_roll_np_type(self):
        output = roll(self.test_np, shift=1, axis=0)
        self.assertIsInstance(output, np.ndarray, 'Output type must be NumPy array')

    def test_roll_np_1d_shift(self):
        output = roll(self.test_np, shift=1, axis=0)
        self.assertTrue(np.array_equal(output, self.test_output_np_1d))

    def test_roll_np_2d_shift(self):
        output = roll(self.test_np, shift=[-1, -1], axis=[0, 1])
        self.assertTrue(np.array_equal(output, self.test_output_np_2d))

    def test_roll_tf_type(self):
        output = roll(self.test_tf, shift=0, axis=1)
        self.assertIsInstance(output, tf.Tensor, 'Output type must be tf.Tensor')

    def test_roll_tf_1d_shift(self):
        output = roll(self.test_tf, shift=1, axis=0)
        self.assertTrue(is_equal(output, self.test_output_tf_1d))

    def test_roll_tf_2d_shift(self):
        output = roll(self.test_tf, shift=[-1, -1], axis=[0, 1])
        self.assertTrue(is_equal(output, self.test_output_tf_2d))

    def test_sign_torch_type(self):
        output = roll(self.test_torch, shift=1, axis=0)
        self.assertIsInstance(output, torch.Tensor, 'Output type must be torch.Tensor')

    def test_roll_torch_1d_shift(self):
        output = roll(self.test_torch, shift=1, axis=0)
        self.assertTrue(is_equal(output, self.test_output_torch_1d))

    def test_roll_torch_2d_shift(self):
        output = roll(self.test_torch, shift=[-1, -1], axis=[0, 1])
        self.assertTrue(is_equal(output, self.test_output_torch_2d))
