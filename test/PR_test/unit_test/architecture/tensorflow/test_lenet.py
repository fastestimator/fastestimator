import unittest

import numpy as np
import tensorflow as tf

from fastestimator.architecture.tensorflow import LeNet


class TestLenet(unittest.TestCase):
    def test_lenet(self):
        data = np.ones((1, 28, 28, 1))
        input_data = tf.constant(data)
        lenet = LeNet()
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 10))

    def test_lenet_class(self):
        data = np.ones((1, 28, 28, 1))
        input_data = tf.constant(data)
        lenet = LeNet(classes=5)
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 5))
