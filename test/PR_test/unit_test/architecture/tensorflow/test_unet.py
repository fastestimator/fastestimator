import unittest

import numpy as np
import tensorflow as tf

from fastestimator.architecture.tensorflow import UNet


class TestUNet(unittest.TestCase):
    def test_unet(self):
        data = np.ones((1, 128, 128, 1))
        input_data = tf.constant(data)
        unet = UNet()
        output_shape = unet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 128, 128, 1))
