import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.loss import Hinge


class TestHinge(unittest.TestCase):
    def test_tf(self):
        true = tf.constant([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                            [1.0, -1.0, -1.0, -1.0]])
        hinge = Hinge(inputs=('x1', 'x2'), outputs='x')
        hinge.build('tf')
        output = hinge.forward(data=[pred, true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.7125))

    def test_torch(self):
        true = torch.tensor([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                             [1.0, -1.0, -1.0, -1.0]])
        hinge = Hinge(inputs=('x1', 'x2'), outputs='x')
        hinge.build('torch')
        output = hinge.forward(data=[pred, true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.7125))
