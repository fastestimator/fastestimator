import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestAbs(unittest.TestCase):
    def test_abs_np_input(self):
        n = np.array([-2, 7, -19])
        obj1 = fe.backend.abs(n)
        obj2 = np.array([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))

    def test_abs_tf_input(self):
        t = tf.constant([-2, 7, -19])
        obj1 = fe.backend.abs(t)
        obj2 = tf.constant([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))

    def test_abs_torch_input(self):
        t = torch.tensor([-2, 7, -19])
        obj1 = fe.backend.abs(t)
        obj2 = torch.tensor([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))
