import unittest
from typing import Any

import numpy as np
import tensorflow as tf
import torch

from fastestimator.test.unittest_util import is_equal


class testIsEqual(unittest.TestCase):
    def test_is_equal_numpy_array(self):
        obj1 = np.array([1, 2.2, -3])
        obj2 = np.array([1, 2.2, -3])
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = np.array([1, 2.2, -3, 4.0])
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = np.array([-1, 2.2, 3])
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_list(self):
        obj1 = []
        obj2 = []
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = [1, 2, 3]
        obj2 = [1, 2, 3]
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = [1, 2, 4]
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = [1, 2, 3, 4]
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_tuple(self):
        obj1 = tuple()
        obj2 = tuple()
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = (1, 2, 3)
        obj2 = (1, 2, 3)
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = (1, 2, 4)
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = (1, 2, 3, 4)
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_set(self):
        obj1 = set()
        obj2 = set()
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = {1, 2, 3}
        obj2 = {1, 2, 3}
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = {1, 2, 4}
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = {1, 2, 3, 4}
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_tf_tensor(self):
        obj1 = tf.constant(1.5)
        obj2 = tf.constant(1.5)

        self.assertTrue(is_equal(obj1, obj2))

        obj2 = tf.constant(0)
        self.assertFalse(is_equal(obj1, obj2))

        obj1 = tf.Variable([1.5, 0, -2.3])
        obj2 = tf.Variable([1.5, 0, -2.3])

        self.assertTrue(is_equal(obj1, obj2))

        obj2 = tf.Variable([1.5, 0, -10])
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = tf.Variable([1.5, 0, -2.3, 100])
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_torch_tensor(self):
        obj1 = torch.Tensor([1.5, 0, -2.3])
        obj2 = torch.Tensor([1.5, 0, -2.3])
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = torch.Tensor([1.5, 0, -10])
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = torch.Tensor([1.5, 0, -2.3, 100])
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_dict_mixture(self):
        obj1 = dict()
        obj2 = dict()
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = {"1": 1, "2": 0, "3": -1.5}
        obj2 = {"1": 1, "2": 0, "3": -1.5}
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = {"10": 1, "2": 0, "3": -1.5}
        self.assertFalse(is_equal(obj1, obj2))

        obj2 = {"1": 1, "2": 0.5, "3": -1.5}
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_type_mixture(self):
        obj1 = [tf.constant([1, 2]), tf.constant([0.1, 0.2])]
        obj2 = [tf.constant([1, 2]), tf.constant([0.1, 0.2])]
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = [tf.constant([1, 2]), tf.constant([0.1, -0.2])]
        self.assertFalse(is_equal(obj1, obj2))

        obj1 = {"1": np.array([-1, 2.5]), "2": {"3": torch.Tensor([1.5])}}
        obj2 = {"1": np.array([-1, 2.5]), "2": {"3": torch.Tensor([1.5])}}
        self.assertTrue(is_equal(obj1, obj2))

        obj2 = {"1": [-1, 2.5], "2": {"3": torch.Tensor([1.5])}}
        self.assertFalse(is_equal(obj1, obj2))

    def test_is_equal_dtype_tf(self):
        obj1 = tf.constant([1, 2], dtype=tf.float32)
        obj2 = tf.constant([1, 2], dtype=tf.float64)
        self.assertFalse(is_equal(obj1, obj2, assert_dtype=True))

    def test_is_equal_dtype_torch(self):
        obj1 = torch.tensor([1, 2], dtype=torch.float32)
        obj2 = torch.tensor([1, 2], dtype=torch.float64)
        self.assertFalse(is_equal(obj1, obj2, assert_dtype=True))

    def test_is_equal_dtype_np(self):
        obj1 = np.array([1, 2], dtype=np.float32)
        obj2 = np.array([1, 2], dtype=np.float64)
        self.assertFalse(is_equal(obj1, obj2, assert_dtype=True))