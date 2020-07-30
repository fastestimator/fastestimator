import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestMaximum(unittest.TestCase):
    def test_maximum_np_input(self):
        n1 = np.array([[2, 7, 6]])
        n2 = np.array([[2, 7, 5]])
        res = fe.backend.maximum(n1, n2)
        self.assertTrue(is_equal(res, n1))

    def test_maximum_tf_input(self):
        t1 = tf.constant([[2, 7, 6]])
        t2 = tf.constant([[2, 7, 5]])
        res = fe.backend.maximum(t1, t2)
        self.assertTrue(is_equal(res, t1))

    def test_maximum_torch_input(self):
        p1 = torch.tensor([[2, 7, 6]])
        p2 = torch.tensor([[2, 7, 5]])
        res = fe.backend.maximum(p1, p2)
        self.assertTrue(is_equal(res, p1))
