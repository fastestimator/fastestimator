import unittest

import tensorflow as tf
import numpy as np
import torch

from fastestimator.op.tensorop.loss.mix_loss import MixLoss
from fastestimator.op.tensorop.loss.cross_entropy import CrossEntropy


class TestMixUpBatch(unittest.TestCase):
    def test_mixup_batch_tf(self):
        true_binary = tf.constant([[1.0], [2.0], [4.0]])
        pred_binary = tf.constant([[1.0], [3.0], [4.5]])

        ml = MixLoss(CrossEntropy(inputs=("y_pred", "y"), mode="train", outputs="loss"), lam="lambda")
        output = ml.forward(data=[0.1, pred_binary, true_binary], state={})

        self.assertTrue(np.allclose(output.numpy(), -20.444319))

    def test_mixup_batch_torch(self):
        true_binary = torch.tensor([[1], [0], [1], [0]])
        pred_binary = torch.tensor([[0.9], [0.3], [0.8], [0.1]])

        ml = MixLoss(CrossEntropy(inputs=("y_pred", "y"), mode="train", outputs="loss"), lam="lambda")
        output = ml.forward(data=[0.1, pred_binary, true_binary], state={})

        self.assertTrue(np.allclose(output.detach().numpy(), 1.6889441))
