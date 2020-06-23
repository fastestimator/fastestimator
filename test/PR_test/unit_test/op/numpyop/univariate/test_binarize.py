import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import Binarize
from fastestimator.test.unittest_util import is_equal


class TestBinarize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = np.array([1, 2, 3, 4])
        cls.single_output = [np.array([1, 1, 1, 1])]
        cls.multi_input = [np.array([2, 2]), np.array([0, 1, 2])]
        cls.multi_output = [np.array([1, 1]), np.array([0, 1, 1])]

    def test_single_input(self):
        op = Binarize(threshold=1, inputs='x', outputs='x')
        data = op.forward(data=[np.array([1, 2, 3, 4])], state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = Binarize(threshold=1, inputs='x', outputs='x')
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))
