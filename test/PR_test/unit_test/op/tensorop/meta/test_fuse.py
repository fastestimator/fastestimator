import unittest

import tensorflow as tf

from fastestimator.op.tensorop.meta import Fuse
from fastestimator.op.tensorop import LambdaOp


class TestFuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_shape = (28, 28, 3)
        cls.multi_input_tf = [tf.random.uniform(shape=(28, 28, 3)), tf.random.uniform(shape=(28, 28, 3))]

    def test_single_input(self):
        a = LambdaOp(inputs='x', outputs='y', fn=lambda x: x + 1, mode='test')
        b = LambdaOp(inputs=['y', 'z'], outputs='w', fn=lambda x, y: x + y, mode='test')
        fuse = Fuse([a, b], repeat=2)
        with self.subTest('Check op inputs'):
            self.assertListEqual(fuse.inputs, ['x', 'z'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(fuse.outputs, ['y', 'w'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(fuse.mode, {'test'})
        output = fuse.forward(data=self.multi_input_tf, state={"mode": "test"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)
