import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.gradient import Watch
from fastestimator.test.unittest_util import is_equal


class TestWatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.Variable([1., 2., 4.])
        cls.tf_output = [tf.Variable([1., 2., 4.], dtype=tf.float32), tf.constant([1, 4, 16], dtype=tf.float32)]
        cls.torch_data = torch.tensor([1., 2., 4.], requires_grad=True)
        cls.torch_output = [torch.tensor([1, 2, 4], dtype=torch.float32), torch.tensor([1, 4, 16], dtype=torch.float32)]

    def test_tf_input(self):
        watch = Watch(inputs='x')
        with tf.GradientTape(persistent=True) as tape:
            x = self.tf_data * self.tf_data
            output = watch.forward(data=[self.tf_data, x], state={'tape': tape})
        self.assertTrue(is_equal(output, self.tf_output))

    def test_torch_input(self):
        watch = Watch(inputs='x')
        x = self.torch_data * self.torch_data
        output = watch.forward(data=[self.torch_data, x], state={'tape': None})
        self.assertTrue(is_equal(output, self.torch_output))
