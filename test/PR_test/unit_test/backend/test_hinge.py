import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestHinge(unittest.TestCase):
    def test_tf(self):
        true = tf.constant([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                            [1.0, -1.0, -1.0, -1.0]])
        b = fe.backend.hinge(y_pred=pred, y_true=true)
        self.assertTrue(is_equal(b, tf.constant([0.8, 1.2, 0.85, 0.0])))

    def test_torch(self):
        true = torch.tensor([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                             [1.0, -1.0, -1.0, -1.0]])
        b = fe.backend.hinge(y_pred=pred, y_true=true)
        self.assertTrue(is_equal(b, torch.tensor([0.8, 1.2, 0.85, 0.0])))
