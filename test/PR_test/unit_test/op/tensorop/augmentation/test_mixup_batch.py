import unittest

import tensorflow as tf
import torch

from fastestimator.op.tensorop.augmentation.mixup_batch import MixUpBatch
from fastestimator.test.unittest_util import is_equal


class MyTFBeta:
    @staticmethod
    def sample(sample_shape=(1, )):
        return 0.5 * tf.ones(shape=sample_shape)


class MyTorchBeta:
    @staticmethod
    def sample(sample_shape=(1, )):
        return 0.5 * torch.ones(size=sample_shape)


class TestMixUpBatch(unittest.TestCase):
    def test_tf_shared_beta(self):
        data = tf.constant([[1.0], [3.0], [4.5]])
        expected = tf.constant([[2.75], [2], [3.75]])
        mu = MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=1.0, mode="train", shared_beta=True)
        mu.build('tf')
        mu.beta = MyTFBeta()
        output = mu.forward(data=[data], state={})
        self.assertTrue(is_equal(output[0], expected))

    def test_tf_different_beta(self):
        data = tf.constant([[1.0], [3.0], [4.5]])
        expected = tf.constant([[2.75], [2], [3.75]])
        mu = MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=1.0, mode="train", shared_beta=False)
        mu.build('tf')
        mu.beta = MyTFBeta()
        output = mu.forward(data=[data], state={})
        self.assertTrue(is_equal(output[0], expected))

    def test_torch_shared_beta(self):
        data = torch.tensor([[1.0], [3.0], [4.5]])
        expected = torch.tensor([[2.75], [2], [3.75]])
        mu = MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=1.0, mode="train", shared_beta=True)
        mu.build('torch')
        mu.beta = MyTorchBeta()
        output = mu.forward(data=[data], state={})
        self.assertTrue(is_equal(output[0], expected))

    def test_torch_different_beta(self):
        data = torch.tensor([[1.0], [3.0], [4.5]])
        expected = torch.tensor([[2.75], [2], [3.75]])
        mu = MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=1.0, mode="train", shared_beta=False)
        mu.build('torch')
        mu.beta = MyTorchBeta()
        output = mu.forward(data=[data], state={})
        self.assertTrue(is_equal(output[0], expected))
