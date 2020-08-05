import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestMatmul(unittest.TestCase):
    def test_np(self):
        a = np.array([[0, 1, 2], [3, 4, 5]])
        b = np.array([[1], [2], [3]])
        c = fe.backend.matmul(a, b)
        self.assertTrue(is_equal(c, np.array([[8], [26]])))

    def test_tf(self):
        a = tf.constant([[0, 1, 2], [3, 4, 5]])
        b = tf.constant([[1], [2], [3]])
        c = fe.backend.matmul(a, b)
        self.assertTrue(is_equal(c, tf.constant([[8], [26]])))

    def test_torch(self):
        a = torch.tensor([[0, 1, 2], [3, 4, 5]])
        b = torch.tensor([[1], [2], [3]])
        c = fe.backend.matmul(a, b)
        self.assertTrue(is_equal(c, torch.tensor([[8], [26]])))
