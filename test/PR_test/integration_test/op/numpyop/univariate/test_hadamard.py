import unittest

import numpy as np

from fastestimator.op.numpyop.univariate.hadamard import Hadamard
from fastestimator.test.unittest_util import is_equal


class TestHadamard(unittest.TestCase):
    def test_4class(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=4)
        output = tohadamard.forward(data=[np.array([3.0, 2.0, 2.0, 2.0, 0.0])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                np.array([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                          [-1., 1., 1., 1.]])))

    def test_4class_single_input(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=4)
        output = tohadamard.forward(data=[0], state={})[0]
        self.assertTrue(is_equal(output, np.array([-1., 1., 1., 1.])))

    def test_10class_16code(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        output = tohadamard.forward(data=[np.array([[0], [1], [9], [0], [4]])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                np.array([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                           1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                 -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                        1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                          [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])))
