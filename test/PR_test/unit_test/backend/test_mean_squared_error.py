import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestMeanSquaredError(unittest.TestCase):
    def test_mean_squared_error_tf_input(self):
        true = tf.constant([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05], [1.0, 0.0, 0.0, 0.0]])
        obj1 = fe.backend.mean_squared_error(y_pred=pred, y_true=true).numpy()
        obj2 = np.array([0.00625, 0.035, 0.01625, 0.0])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_mean_squared_error_torch_input(self):
        true = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05], [1.0, 0.0, 0.0,
                                                                                                    0.0]])
        obj1 = fe.backend.mean_squared_error(y_pred=pred, y_true=true).numpy()
        obj2 = np.array([0.00625, 0.035, 0.01625, 0.0])
        self.assertTrue(np.allclose(obj1, obj2))
