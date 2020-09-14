import unittest

import numpy as np

from fastestimator.op.numpyop import LambdaOp
from fastestimator.test.unittest_util import is_equal


class TestLambdaOp(unittest.TestCase):
    def test_single_input(self):
        op = LambdaOp(fn=np.sum)
        data = op.forward(data=[[1, 2, 3]], state={})
        self.assertEqual(data, 6)

    def test_multi_input(self):
        op = LambdaOp(fn=np.reshape)
        data = op.forward(data=[np.array([1, 2, 3, 4]), (2, 2)], state={})
        self.assertTrue(is_equal(data, np.array([[1, 2], [3, 4]])))
