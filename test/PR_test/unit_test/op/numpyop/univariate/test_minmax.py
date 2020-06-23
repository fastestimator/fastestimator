import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import Minmax
from fastestimator.test.unittest_util import is_equal


class TestMinmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [np.array([1, 2, 3, 5])]
        cls.single_output = [np.array([0, 0.25, 0.5, 1])]
        cls.multi_input = [np.array([2, 2]), np.array([0, 1, 2])]
        cls.multi_output = [np.array([0, 0]), np.array([0, 0.5, 1])]

    def test_single_input(self):
        op = Minmax(inputs='x', outputs='x')
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = Minmax(inputs='x', outputs='x')
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))
