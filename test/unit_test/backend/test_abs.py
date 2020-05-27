import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestAbs(unittest.TestCase):
    def test_abs_np_input(self):
        n = np.array([-2, 7, -19])
        b = fe.backend.abs(n)
        self.assertTrue(np.array_equal(b, np.array([2, 7, 19])))

    def test_abs_tf_input(self):
        t = tf.constant([-2, 7, -19])
        b = fe.backend.abs(t)
        self.assertTrue(np.array_equal(b, np.array([2, 7, 19])))

    def test_abs_torch_input(self):
        p = torch.tensor([-2, 7, -19])
        b = fe.backend.abs(p)
        self.assertTrue(np.array_equal(b, np.array([2, 7, 19])))
