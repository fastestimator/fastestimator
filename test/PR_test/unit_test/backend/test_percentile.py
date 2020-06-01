import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestPercentile(unittest.TestCase):
    def test_percentile_tf_input_axis_none(self):
        t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50)
        obj2 = tf.constant([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50])
        obj2 = tf.constant([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_tf_input_axis_not_none(self):
        t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50, axis=0)
        obj2 = tf.constant([[2, 4, 6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=1)
        obj2 = tf.constant([[3], [5], [6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1])
        obj2 = tf.constant([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1])
        obj2 = tf.constant([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_tf_input_axis_not_none_keepdims_false(self):
        t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50, axis=0, keepdims=False)
        obj2 = tf.constant([2, 4, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=1, keepdims=False)
        obj2 = tf.constant([3, 5, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1], keepdims=False)
        obj2 = tf.constant(5)
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1], keepdims=False)
        obj2 = tf.constant([1, 5])
        self.assertTrue(is_equal(obj1, obj2))

    # ------------------------- torch input --------------------------------------
    def test_percentile_torch_input_axis_none(self):
        t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50)
        obj2 = torch.tensor([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50])
        obj2 = torch.tensor([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_torch_input_axis_not_none(self):
        t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50, axis=0)
        obj2 = torch.tensor([[2, 4, 6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=1)
        obj2 = torch.tensor([[3], [5], [6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1])
        obj2 = torch.tensor([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1])
        obj2 = torch.tensor([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_torch_input_axis_not_none_keepdims_false(self):
        t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(t, percentiles=50, axis=0, keepdims=False)
        obj2 = torch.tensor([2, 4, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=1, keepdims=False)
        obj2 = torch.tensor([3, 5, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1], keepdims=False)
        obj2 = torch.tensor(5)
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1], keepdims=False)
        obj2 = torch.tensor([1, 5])
        self.assertTrue(is_equal(obj1, obj2))

    # ------------------------- numpy input ---------------------------------------
    def test_percentile_np_input_axis_none(self):
        n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(n, percentiles=50)
        obj2 = np.array([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=[0, 50])
        obj2 = np.array([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_np_input_axis_not_none(self):
        n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(n, percentiles=50, axis=0)
        obj2 = np.array([[2, 4, 6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=50, axis=1)
        obj2 = np.array([[3], [5], [6]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=50, axis=[0, 1])
        obj2 = np.array([[5]])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=[0, 50], axis=[0, 1])
        obj2 = np.array([[[1]], [[5]]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_np_input_axis_not_none_keepdims_false(self):
        n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
        obj1 = fe.backend.percentile(n, percentiles=50, axis=0, keepdims=False)
        obj2 = np.array([2, 4, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=50, axis=1, keepdims=False)
        obj2 = np.array([3, 5, 6])
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=50, axis=[0, 1], keepdims=False)
        obj2 = np.int64(5)
        self.assertTrue(is_equal(obj1, obj2))

        obj1 = fe.backend.percentile(n, percentiles=[0, 50], axis=[0, 1], keepdims=False)
        obj2 = np.array([1, 5])
        self.assertTrue(is_equal(obj1, obj2))
