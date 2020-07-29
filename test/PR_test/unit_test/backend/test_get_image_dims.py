import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import get_image_dims
from fastestimator.test.unittest_util import is_equal


class TestGetImageDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.random.random((2, 12, 12, 3))
        cls.test_tf = tf.random.uniform((2, 12, 12, 3))
        cls.test_torch = torch.rand((2, 3, 12, 12))
        cls.test_output = (3, 12, 12)

    def test_np_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_np), self.test_output))

    def test_tf_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_tf), self.test_output))

    def test_torch_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_torch), self.test_output))
