import unittest

import tensorflow as tf
import tensorflow_probability as tfp

import fastestimator as fe


class TestInstanceNorm(unittest.TestCase):
    def test_instance_norm(self):
        n = tfp.distributions.Normal(loc=10, scale=2)
        x = n.sample(sample_shape=(1, 100, 100, 1))
        m = fe.layers.tensorflow.InstanceNormalization()
        y = m(x)
        self.assertLess(tf.reduce_mean(y), 0.1)
        self.assertLess(tf.math.reduce_std(y), 0.1)
