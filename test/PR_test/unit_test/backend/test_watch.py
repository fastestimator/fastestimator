import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestWatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = np.array([[0, 1], [2, 3]])
        cls.data_tf = tf.constant([[0, 1], [2, 3]])
        cls.data_torch = torch.tensor([[0, 1], [2, 3]])
        cls.op = np.array([[0, 0], [0, 0]])
        cls.op_tf = tf.zeros((2, 2), dtype=tf.int32)
        cls.op_torch = torch.tensor([[0, 0], [0, 0]])

    def test_zeros_like_np(self):
        self.assertTrue(fet.is_equal(fe.backend.zeros_like(self.data_np), self.op))

    def test_zeros_like_tf(self):
        self.assertTrue(fet.is_equal(fe.backend.zeros_like(self.data_tf), self.op_tf))

    def test_zeros_like_torch(self):
        self.assertTrue(fet.is_equal(fe.backend.zeros_like(self.data_torch), self.op_torch))
