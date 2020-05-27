import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestArgmax(unittest.TestCase):
    def test_argmax_np_input(self):
        n = np.array([[2, 7, 5], [9, 1, 3], [4, 8, 2]])
        b = fe.backend.argmax(n, axis=0)  # [1, 2, 0]
        self.assertTrue(np.array_equal(b, np.array([1, 2, 0])))
        b = fe.backend.argmax(n, axis=1)  # [1, 0, 1]
        self.assertTrue(np.array_equal(b, np.array([1, 0, 1])))

    def test_argmax_tf_input(self):
        t = tf.constant([[2, 7, 5], [9, 1, 3], [4, 8, 2]])
        b = fe.backend.argmax(t, axis=0)  # [1, 2, 0]
        self.assertTrue(np.array_equal(b, np.array([1, 2, 0])))
        b = fe.backend.argmax(t, axis=1)  # [1, 0, 1]
        self.assertTrue(np.array_equal(b, np.array([1, 0, 1])))

    def test_argmax_torch_input(self):
        p = torch.tensor([[2, 7, 5], [9, 1, 3], [4, 8, 2]])
        b = fe.backend.argmax(p, axis=0)  # [1, 2, 0]
        self.assertTrue(np.array_equal(b, np.array([1, 2, 0])))
        b = fe.backend.argmax(p, axis=1)  # [1, 0, 1]
        self.assertTrue(np.array_equal(b, np.array([1, 0, 1])))
