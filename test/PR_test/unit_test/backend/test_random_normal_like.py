import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import random_normal_like


class TestRandomNormalLike(unittest.TestCase):
    def test_random_normal_np_input(self):
        test = np.array([[0, 1], [1, 1]])
        output = random_normal_like(test)
        # check the type of output array
        self.assertIsInstance(output, np.ndarray, 'Output must be NumPy Array')
        # check if the shape is same
        self.assertTrue((output.shape == (2, 2)), 'Output array shape should be same as input')

    def test_random_normal_tf_input(self):
        test = tf.constant([[0, 1], [2, 2]])
        output_shape = tf.shape([2, 2])
        output = random_normal_like(test)
        # check the type of output array
        self.assertIsInstance(output, tf.Tensor, 'Output type must be tf.Tensor')
        # check if the shape is same
        self.assertTrue(tf.reduce_all(tf.equal(tf.shape(output), output_shape)),
                        'Output tensor shape should be same as input')

    def test_random_normal_torch_input(self):
        test = torch.Tensor([[1, 1], [2, 3]])
        output_shape = (2, 2)
        output = random_normal_like(test)
        # check the type of output array
        self.assertIsInstance(output, torch.Tensor, 'Output must be torch.Tensor')
        # check if the shape is same
        self.assertTrue((output.size() == output_shape), 'Output tensor shape should be same as input')
