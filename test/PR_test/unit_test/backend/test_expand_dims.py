import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestExpandDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np_input = np.array([2, 7, 5])
        cls.torch_input = torch.tensor([2, 7, 5])
        cls.tf_input = tf.constant([2, 7, 5])

    def test_expand_dims_np_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.np_input, axis=0)
        obj2 = np.array([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_np_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.np_input, axis=1)
        obj2 = np.array([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_tf_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.tf_input, axis=0)
        obj2 = tf.constant([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_tf_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.tf_input, axis=1)
        obj2 = tf.constant([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_torch_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.torch_input, axis=0)
        obj2 = torch.tensor([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_torch_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.torch_input, axis=1)
        obj2 = torch.tensor([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))
