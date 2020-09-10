import unittest

import tensorflow as tf
import torch

from fastestimator.op.tensorop import LambdaOp
from fastestimator.op.tensorop.meta import Sometimes


class TestSometimes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_shape = (28, 28, 3)
        cls.single_input_tf = [tf.random.uniform(shape=(28, 28, 3))]
        cls.multi_input_tf = [tf.random.uniform(shape=(28, 28, 3)), tf.random.uniform(shape=(28, 28, 3))]
        cls.single_input_torch = [torch.ones(size=(28, 28, 3))]
        cls.multi_input_torch = [torch.ones(size=(28, 28, 3)), torch.ones(size=(28, 28, 3))]

    def test_single_input_tf(self):
        a = LambdaOp(inputs='x', outputs='x', fn=lambda x: x + 1)
        sometimes = Sometimes(a, prob=0.75)
        sometimes.build('tf')
        output = sometimes.forward(data=self.single_input_tf, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)

    def test_multi_input_tf(self):
        a = LambdaOp(inputs=['x', 'y'], outputs=['y', 'x'], fn=lambda x, y: [x + y, x - y])
        sometimes = Sometimes(a)
        sometimes.build('tf')
        output = sometimes.forward(data=self.multi_input_tf, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output in output:
            with self.subTest('Check output image shape'):
                self.assertEqual(img_output.shape, self.output_shape)

    def test_single_input_torch(self):
        a = LambdaOp(inputs='x', outputs='x', fn=lambda x: x + 1)
        sometimes = Sometimes(a, prob=0.75)
        sometimes.build('torch')
        output = sometimes.forward(data=self.single_input_torch, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)

    def test_multi_input_torch(self):
        a = LambdaOp(inputs=['x', 'y'], outputs=['x', 'y'], fn=lambda x, y: [x + y, x - y])
        sometimes = Sometimes(a)
        sometimes.build('torch')
        output = sometimes.forward(data=self.multi_input_torch, state={})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output list length'):
            self.assertEqual(len(output), 2)
        for img_output in output:
            with self.subTest('Check output image shape'):
                self.assertEqual(img_output.shape, self.output_shape)
