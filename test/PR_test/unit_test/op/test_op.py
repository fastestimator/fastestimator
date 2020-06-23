import unittest

import numpy as np

from fastestimator.op.op import LambdaOp, Op, get_inputs_by_op, write_outputs_by_op
from fastestimator.test.unittest_util import is_equal


class TestOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.testop = Op(inputs="x", outputs="y", mode="train")

    def test_op_initialization(self):
        self.assertEqual(self.testop.inputs, ["x"])
        self.assertEqual(self.testop.outputs, ["y"])
        self.assertEqual(self.testop.mode, {"train"})


class TestGetInputsByOp(unittest.TestCase):
    def test_single_key_single_data(self):
        data = get_inputs_by_op(op=Op(inputs="x"), store={"x": 1})
        self.assertEqual(data, 1)

    def test_single_key_multi_data(self):
        data = get_inputs_by_op(op=Op(inputs="x"), store={"x": [1, 2]})
        self.assertEqual(data, [1, 2])

    def test_multi_key_multi_data(self):
        x, y = get_inputs_by_op(op=Op(inputs=["x", "y"]), store={"x": 1, "y": [1, 2]})
        self.assertEqual(x, 1)
        self.assertEqual(y, [1, 2])


class TestWriteOutputsByOp(unittest.TestCase):
    def test_single_key_single_data(self):
        batch = {}
        write_outputs_by_op(op=Op(outputs="x"), store=batch, outputs=1)
        self.assertEqual(batch, {"x": 1})

    def test_single_key_multi_data(self):
        batch = {}
        write_outputs_by_op(op=Op(outputs="x"), store=batch, outputs=[1, 2])
        self.assertEqual(batch, {"x": [1, 2]})

    def test_multi_key_multi_data(self):
        batch = {}
        write_outputs_by_op(op=Op(outputs=["x", "y"]), store=batch, outputs=[1, [1, 2]])
        self.assertEqual(batch, {"x": 1, "y": [1, 2]})
    

class TestLambdaOp(unittest.TestCase):
    def test_single_input(self):
        op = LambdaOp(fn=np.sum)
        data = op.forward(data=[[1, 2, 3]], state={})
        self.assertEqual(data, 6)

    def test_multi_input(self):
        op = LambdaOp(fn=np.reshape)
        data = op.forward(data=[np.array([1, 2, 3, 4]), (2, 2)], state={})
        self.assertTrue(is_equal(data, np.array([[1, 2], [3, 4]])))
