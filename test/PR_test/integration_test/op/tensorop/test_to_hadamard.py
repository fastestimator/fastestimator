import unittest

import tensorflow as tf
import torch

from fastestimator.op.tensorop.to_hadamard import ToHadamard
from fastestimator.test.unittest_util import is_equal


class TestToHadamard(unittest.TestCase):
    def test_tf_4class(self):
        tohadamard = ToHadamard(inputs='y', outputs='y', n_classes=4)
        tohadamard.build('tf')
        output = tohadamard.forward(data=[tf.constant([3.0, 2.0, 2.0, 2.0, 0.0])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                tf.constant([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                             [-1., 1., 1., 1.]])))

    def test_tf_10class_16code(self):
        tohadamard = ToHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        tohadamard.build('tf')
        output = tohadamard.forward(data=[tf.constant([[0], [1], [9], [0], [4]])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                tf.constant([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                              1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                    -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                           1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])))

    def test_torch_4class(self):
        tohadamard = ToHadamard(inputs='y', outputs='y', n_classes=4)
        tohadamard.build('torch')
        output = tohadamard.forward(data=[torch.tensor([3.0, 2.0, 2.0, 2.0, 0.0])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                torch.tensor([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                              [-1., 1., 1., 1.]])))

    def test_torch_10class_16code(self):
        tohadamard = ToHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        tohadamard.build('torch')
        output = tohadamard.forward(data=[torch.tensor([[0], [1], [9], [0], [4]])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                torch.tensor([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                               1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                     -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                            1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                              [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])))
