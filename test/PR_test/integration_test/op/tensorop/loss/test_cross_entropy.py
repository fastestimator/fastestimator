import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.test.unittest_util import is_equal

class TestCrossEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # binary ce
        cls.tf_true_binary = tf.constant([[1.0], [2.0], [4.0]])
        cls.tf_pred_binary = tf.constant([[1.0], [3.0], [4.5]])
        # torch binary ce
        cls.torch_true_binary = torch.tensor([[1], [0], [1], [0]])
        cls.torch_pred_binary = torch.tensor([[0.9], [0.3], [0.8], [0.1]])
        # categorical ce
        cls.tf_true_cat = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cls.tf_pred_cat = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        # sparse categorical ce
        cls.tf_true_sparse = tf.constant([[0], [1], [0]])
        cls.tf_pred_sparse = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])

    def test_binary_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.tf_pred_binary, self.tf_true_binary], state={})
        self.assertTrue(np.allclose(output.numpy(), -20.444319))

    def test_categorical_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.tf_pred_cat, self.tf_true_cat], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.22839302))

    def test_sparse_categorical_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.tf_pred_sparse, self.tf_true_sparse], state={})
        self.assertTrue(np.allclose(output.numpy(), 2.5336342))

    def test_torch_input(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.torch_pred_binary, self.torch_true_binary], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.1976349))


if __name__ == "__main__":
    unittest.main()
