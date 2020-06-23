import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import ToArray
from fastestimator.test.unittest_util import is_equal


class TestToArray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = [1, 2, 3]
        cls.output = [np.array(1), np.array(2), np.array(3)]

    def test_output_values(self):
        op = ToArray(inputs='x', outputs='x')
        data = op.forward(data=self.input, state={})
        self.assertTrue(is_equal(data, self.output))
