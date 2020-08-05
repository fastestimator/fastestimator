import unittest

import tensorflow as tf
import torch

from fastestimator.op.tensorop.reshape import Reshape
from fastestimator.test.unittest_util import is_equal


class TestReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.constant([[1., 2., 4.], [1., 2., 6.]])
        cls.tf_output = tf.constant([[1., 2., 4., 1., 2., 6.]])
        cls.torch_data = torch.tensor([[1., 2., 4.], [1., 2., 6.]])
        cls.torch_output = torch.tensor([[1., 2., 4., 1., 2., 6.]])

    def test_tf_input(self):
        reshape = Reshape(inputs='x', outputs='x', shape=(1, 6))
        output = reshape.forward(data=[self.tf_data], state={})
        self.assertTrue(is_equal(output[0], self.tf_output))

    def test_torch_input(self):
        reshape = Reshape(inputs='x', outputs='x', shape=(1, 6))
        output = reshape.forward(data=[self.torch_data], state={})
        self.assertTrue(is_equal(output[0], self.torch_output))
