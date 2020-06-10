import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestPercentile(unittest.TestCase):
    def test_percentile_tf_input_axis_none(self):
        with self.subTest("even_elements"):
            t = tf.constant([1, 2])
            obj1 = fe.backend.percentile(t, percentiles=50)
            obj2 = tf.constant([1])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("two_dimensional"):
            t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50)
            obj2 = tf.constant([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50])
            obj2 = tf.constant([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_tf_input_axis_not_none(self):
        with self.subTest("two_dimensional"):
            t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50, axis=0)
            obj2 = tf.constant([[2, 4, 6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=1)
            obj2 = tf.constant([[3], [5], [6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1])
            obj2 = tf.constant([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1])
            obj2 = tf.constant([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_tf_input_axis_not_none_keepdims_false(self):
        with self.subTest("two_dimensional"):
            t = tf.constant([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50, axis=0, keepdims=False)
            obj2 = tf.constant([2, 4, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=1, keepdims=False)
            obj2 = tf.constant([3, 5, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1], keepdims=False)
            obj2 = tf.constant(5)
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1], keepdims=False)
            obj2 = tf.constant([1, 5])
            self.assertTrue(is_equal(obj1, obj2))

    # ------------------------- torch input --------------------------------------
    def test_percentile_torch_input_axis_none(self):
        with self.subTest("even_elements"):
            t = torch.tensor([1, 2])
            obj1 = fe.backend.percentile(t, percentiles=50)
            obj2 = torch.tensor([1])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("two_dimensional"):
            t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50)
            obj2 = torch.tensor([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50])
            obj2 = torch.tensor([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_torch_input_axis_not_none(self):
        with self.subTest("two_dimensional"):
            t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50, axis=0)
            obj2 = torch.tensor([[2, 4, 6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=1)
            obj2 = torch.tensor([[3], [5], [6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1])
            obj2 = torch.tensor([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1])
            obj2 = torch.tensor([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_torch_input_axis_not_none_keepdims_false(self):
        with self.subTest("two_dimensional"):
            t = torch.tensor([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(t, percentiles=50, axis=0, keepdims=False)
            obj2 = torch.tensor([2, 4, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=1, keepdims=False)
            obj2 = torch.tensor([3, 5, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(t, percentiles=50, axis=[0, 1], keepdims=False)
            obj2 = torch.tensor(5)
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(t, percentiles=[0, 50], axis=[0, 1], keepdims=False)
            obj2 = torch.tensor([1, 5])
            self.assertTrue(is_equal(obj1, obj2))

    # ------------------------- numpy input ---------------------------------------
    def test_percentile_np_input_axis_none(self):
        with self.subTest("even_elements"):
            n = np.array([1, 2])
            obj1 = fe.backend.percentile(n, percentiles=50)
            obj2 = np.array([1])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("two_dimensional"):
            n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(n, percentiles=50)
            obj2 = np.array([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(n, percentiles=[0, 50])
            obj2 = np.array([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_np_input_axis_not_none(self):
        with self.subTest("two_dimensional"):
            n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(n, percentiles=50, axis=0)
            obj2 = np.array([[2, 4, 6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(n, percentiles=50, axis=1)
            obj2 = np.array([[3], [5], [6]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(n, percentiles=50, axis=[0, 1])
            obj2 = np.array([[5]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(n, percentiles=[0, 50], axis=[0, 1])
            obj2 = np.array([[[1]], [[5]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_percentile_np_input_axis_not_none_keepdims_false(self):
        with self.subTest("two_dimensional"):
            n = np.array([[1, 3, 9], [2, 7, 5], [8, 4, 6]])
            obj1 = fe.backend.percentile(n, percentiles=50, axis=0, keepdims=False)
            obj2 = np.array([2, 4, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("single_axis"):
            obj1 = fe.backend.percentile(n, percentiles=50, axis=1, keepdims=False)
            obj2 = np.array([3, 5, 6])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("multi_axis"):
            obj1 = fe.backend.percentile(n, percentiles=50, axis=[0, 1], keepdims=False)
            self.assertTrue(is_equal(obj1, 5, assert_type=False))

        with self.subTest("multi_percentile"):
            obj1 = fe.backend.percentile(n, percentiles=[0, 50], axis=[0, 1], keepdims=False)
            obj2 = np.array([1, 5])
            self.assertTrue(is_equal(obj1, obj2))
