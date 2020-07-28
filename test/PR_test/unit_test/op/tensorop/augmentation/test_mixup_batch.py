import unittest

import tensorflow as tf

from fastestimator.op.tensorop.augmentation import MixUpBatch
from fastestimator.test.unittest_util import is_equal


class MyBeta:
    def sample(self):
        return 0.5


class TestMixUpBatch(unittest.TestCase):
    def test_mixup_batch(self):
        data = tf.constant([[1.0], [3.0], [4.5]])
        expected = tf.constant([[2.75], [2], [3.75]])

        mu = MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=1.0, mode="train", shared_beta=True)
        mu.beta = MyBeta()
        output = mu.forward(data=[data], state={})

        self.assertTrue(is_equal(output[0], expected))
