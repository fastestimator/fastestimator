import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.loss import MeanSquaredError


class TestMeanSquaredError(unittest.TestCase):
    def test_tf(self):
        tf_true = tf.constant([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        tf_pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05],
                               [1.0, 0.0, 0.0, 0.0]])
        mse = MeanSquaredError(inputs='x', outputs='x')
        output = mse.forward(data=[tf_pred, tf_true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.014375001))

    def test_torch(self):
        torch_true = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        torch_pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05],
                                   [1.0, 0.0, 0.0, 0.0]])
        mse = MeanSquaredError(inputs='x', outputs='x')
        output = mse.forward(data=[torch_pred, torch_true], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.014375001))
