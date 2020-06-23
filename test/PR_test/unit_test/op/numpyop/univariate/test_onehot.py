import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import Onehot
from fastestimator.test.unittest_util import is_equal


class TestOnehot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_input = [[1], [2], [3], [3]]
        cls.single_output = [
            np.array([0., 1., 0., 0.]),
            np.array([0., 0., 1., 0.]),
            np.array([0., 0., 0., 1.]),
            np.array([0., 0., 0., 1.])
        ]

    def test_input_labels(self):
        op = Onehot(inputs='x', outputs='x', num_classes=4)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))
