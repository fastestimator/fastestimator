import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.argmax import Argmax
from fastestimator.test.unittest_util import is_equal


class TestArgmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.constant([1., 2., 4.])
        cls.tf_output = [tf.convert_to_tensor(2, dtype=tf.int64)]
        cls.torch_data = torch.tensor([1., 2., 4.], requires_grad=True)
        cls.torch_output = [torch.tensor(2, dtype=torch.long)]

    def test_tf_input(self):
        argmax = Argmax(inputs='x', outputs='x')
        output = argmax.forward(data=[self.tf_data], state={})
        self.assertTrue(is_equal(output, self.tf_output))

    def test_torch_input(self):
        argmax = Argmax(inputs='x', outputs='x')
        output = argmax.forward(data=[self.torch_data], state={})
        self.assertTrue(is_equal(output, self.torch_output))
