import unittest

import tensorflow as tf
import numpy as np

from fastestimator.op.tensorop.loss import MixLoss, CrossEntropy


class TestMixUpBatch(unittest.TestCase):
    def test_mixup_batch(self):
        true_binary = tf.constant([[1.0], [2.0], [4.0]])
        pred_binary = tf.constant([[1.0], [3.0], [4.5]])

        ml = MixLoss(CrossEntropy(inputs=("y_pred", "y"), mode="train", outputs="loss"), lam="lambda")
        output = ml.forward(data=[0.1, pred_binary, true_binary], state={})

        self.assertTrue(np.allclose(output[0].numpy(), -20.444319))
