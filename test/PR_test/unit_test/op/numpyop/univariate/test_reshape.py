import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import Reshape
from fastestimator.test.unittest_util import is_equal


class TestReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.array([1, 2, 3, 4])]
        cls.single_output = [np.array([[1, 2], [3, 4]])]
        cls.multi_input = [np.array([2, 2]), np.array([1, 2])]
        cls.multi_output = [np.array([[2, 2]]), np.array([[1, 2]])]

    def test_single_input(self):
        op = Reshape(inputs='x', outputs='x', shape=(2, 2))
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = Reshape(inputs='x', outputs='x', shape=(1, 2))
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))
