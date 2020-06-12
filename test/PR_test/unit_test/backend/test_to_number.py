import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestToNumber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = np.array([1, 2, 3])
        cls.t = tf.constant([1, 2, 3])
        cls.p = torch.tensor([1, 2, 3])

    def test_to_number_np_value(self):
        self.assertTrue(np.allclose(fe.backend.to_number(self.n), self.n))

    def test_to_number_np_type(self):
        self.assertEqual(type(fe.backend.to_number(self.n)), np.ndarray)

    def test_to_number_tf_value(self):
        self.assertTrue(np.allclose(fe.backend.to_number(self.t), self.n))

    def test_to_number_tf_type(self):
        self.assertEqual(type(fe.backend.to_number(self.t)), np.ndarray)

    def test_to_number_torch_value(self):
        self.assertTrue(np.allclose(fe.backend.to_number(self.p), self.n))

    def test_to_number_torch_type(self):
        self.assertEqual(type(fe.backend.to_number(self.p)), np.ndarray)
