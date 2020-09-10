import unittest

import tensorflow as tf

from fastestimator.op.tensorop import LambdaOp
from fastestimator.test.unittest_util import is_equal


class TestLambdaOp(unittest.TestCase):
    def test_single_input(self):
        op = LambdaOp(fn=tf.reduce_sum)
        data = op.forward(data=tf.convert_to_tensor([[1, 2, 3]]), state={})
        self.assertEqual(data, 6)

    def test_multi_input(self):
        op = LambdaOp(fn=tf.reshape)
        data = op.forward(data=[tf.convert_to_tensor([1, 2, 3, 4]), (2, 2)], state={})
        self.assertTrue(is_equal(data, tf.convert_to_tensor([[1, 2], [3, 4]])))
