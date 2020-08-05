import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestGather(unittest.TestCase):
    def test_np_input_3x2(self):
        ind = np.array([1, 0, 1])
        n = np.array([[0, 1], [2, 3], [4, 5]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, np.array([[2, 3], [0, 1], [2, 3]])))

    def test_np_input_3x2x2(self):
        ind = np.array([[1], [0], [1]])
        n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, np.array([[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]])))

    def test_tf_input_3x2(self):
        ind = tf.constant([1, 0, 1])
        n = tf.constant([[0, 1], [2, 3], [4, 5]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, tf.constant([[2, 3], [0, 1], [2, 3]])))

    def test_tf_input_3x2x2(self):
        ind = tf.constant([[1], [0], [1]])
        n = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, tf.constant([[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]])))

    def test_torch_input_3x2(self):
        ind = torch.tensor([1, 0, 1])
        n = torch.tensor([[0, 1], [2, 3], [4, 5]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, torch.tensor([[2, 3], [0, 1], [2, 3]])))

    def test_torch_input_3x2x2(self):
        ind = torch.tensor([[1], [0], [1]])
        n = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        b = fe.backend.gather(n, ind)
        self.assertTrue(is_equal(b, torch.tensor([[[4, 5], [6, 7]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]])))
