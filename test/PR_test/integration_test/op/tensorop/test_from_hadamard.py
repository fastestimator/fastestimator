import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.from_hadamard import FromHadamard
from fastestimator.test.unittest_util import is_equal


class TestFromHadamard(unittest.TestCase):
    def test_tf_4class(self):
        fromhadamard = FromHadamard(inputs='y', outputs='y', n_classes=4)
        fromhadamard.build('tf')
        output = fromhadamard.forward(
            data=[
                tf.constant([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                             [-1., 1., 1., 1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, np.array([3, 2, 2, 2, 0])))

    def test_tf_10class_16code(self):
        fromhadamard = FromHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        fromhadamard.build('tf')
        output = fromhadamard.forward(
            data=[
                tf.constant([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                              1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                    -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                           1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, np.array([0, 1, 9, 0, 4])))

    def test_torch_4class(self):
        fromhadamard = FromHadamard(inputs='y', outputs='y', n_classes=4)
        fromhadamard.build('torch')
        output = fromhadamard.forward(
            data=[
                torch.tensor([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                              [-1., 1., 1., 1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, torch.tensor([3, 2, 2, 2, 0])))

    def test_torch_10class_16code(self):
        fromhadamard = FromHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        fromhadamard.build('torch')
        output = fromhadamard.forward(
            data=[
                torch.tensor([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                               1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                     -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                            1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                              [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, torch.tensor([0, 1, 9, 0, 4])))
