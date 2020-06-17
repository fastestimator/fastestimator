import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestGetGradient(unittest.TestCase):
    def test_get_gradient_tf_tensor_higher_order_false(self):
        x = tf.Variable([1.0, 2.0, 3.0])
        with tf.GradientTape(persistent=True) as tape:
            y = x * x
            with self.subTest("check gradient"):
                obj1 = fe.backend.get_gradient(target=y, sources=x, tape=tape)  # [2.0, 4.0, 6.0]
                obj2 = tf.constant([2.0, 4.0, 6.0])
                self.assertTrue(is_equal(obj1, obj2))

            with self.subTest("check gradient of gradient"):
                obj1 = fe.backend.get_gradient(target=obj1, sources=x, tape=tape)  # None
                self.assertTrue(is_equal(obj1, None))

    def test_get_gradient_tf_tensor_higher_order_true(self):
        x = tf.Variable([1.0, 2.0, 3.0])
        with tf.GradientTape(persistent=True) as tape:
            y = x * x
            with self.subTest("check gradient"):
                obj1 = fe.backend.get_gradient(target=y, sources=x, tape=tape, higher_order=True)  # [2.0, 4.0, 6.0]
                obj2 = tf.constant([2.0, 4.0, 6.0])
                self.assertTrue(is_equal(obj1, obj2))

            with self.subTest("check gradient of gradient"):
                obj1 = fe.backend.get_gradient(target=obj1, sources=x, tape=tape)  # [2.0, 2.0, 2.0]
                obj2 = tf.constant([2.0, 2.0, 2.0])
                self.assertTrue(is_equal(obj1, obj2))

    def test_get_gradient_torch_tensor_higher_order_false(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        obj1 = fe.backend.get_gradient(target=y, sources=x)  # [2.0, 4.0, 6.0]
        obj2 = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(is_equal(obj1, obj2))

    def test_get_gradient_torch_tensor_higher_order_true(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        with self.subTest("check gradient"):
            obj1 = fe.backend.get_gradient(target=y, sources=x, higher_order=True)  # [2.0, 4.0, 6.0]
            obj2 = torch.tensor([2.0, 4.0, 6.0])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("check gradient of gradient"):
            obj1 = fe.backend.get_gradient(target=obj1, sources=x)  # [2.0, 2.0, 2.0]
            obj2 = torch.tensor([2.0, 2.0, 2.0])
            self.assertTrue(is_equal(obj1, obj2))
