import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestClipByValue(unittest.TestCase):
    def test_clip_by_value_tf_input(self):
        t = tf.constant([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=-2, max_value=3)
        obj2 = tf.constant([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_torch_input(self):
        t = torch.tensor([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=-2, max_value=3)
        obj2 = torch.tensor([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_np_input(self):
        t = np.array([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=-2, max_value=3)
        obj2 = np.array([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_tensor_min_max_value_tf(self):
        t = tf.constant([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=tf.constant(-2), max_value=3)
        obj2 = tf.constant([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_tensor_min_max_value_list_tf(self):
        t = tf.constant([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=tf.constant([-2]), max_value=3)
        obj2 = tf.constant([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_tensor_min_max_value_torch(self):
        t = torch.tensor([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=-2, max_value=torch.tensor(3))
        obj2 = torch.tensor([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))

    def test_clip_by_value_tensor_min_max_value_list_torch(self):
        t = torch.tensor([-5, 4, 2, 0, 9, -2])
        obj1 = fe.backend.clip_by_value(t, min_value=-2, max_value=torch.tensor([3]))
        obj2 = torch.tensor([-2, 3, 2, 0, 3, -2])
        self.assertTrue(is_equal(obj1, obj2))
