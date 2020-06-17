import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model


class TestFeedForward(unittest.TestCase):
    def test_feed_forward_tf(self):
        model = one_layer_tf_model()
        x = tf.constant([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        obj1 = fe.backend.feed_forward(model, x)
        obj2 = tf.constant([[6.0], [-2.5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_feed_forward_torch(self):
        model = OneLayerTorchModel()
        x = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        obj1 = fe.backend.feed_forward(model, x).detach()
        obj2 = torch.tensor([[6.0], [-2.5]])
        self.assertTrue(is_equal(obj1, obj2))
