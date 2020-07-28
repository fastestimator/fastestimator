import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import random_mix_patch
from fastestimator.test.unittest_util import is_equal


class TestRandomMixPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_tf = tf.random.uniform((4, 32, 32, 3))
        cls.tf_x = tf.constant([0.25])
        cls.tf_y = tf.constant([0.25])
        cls.tf_lam = tf.constant([0.5])
        cls.tf_output = (tf.constant([0]), tf.constant([19]), tf.constant([0]), tf.constant([19]), 32, 32)
        cls.test_torch = torch.rand((4, 3, 32, 32))
        cls.torch_x = torch.Tensor([0.25])
        cls.torch_y = torch.Tensor([0.25])
        cls.torch_lam = torch.Tensor([0.5])
        cls.torch_output = (torch.Tensor([0]), torch.Tensor([19]), torch.Tensor([0]), torch.Tensor([19]), 32, 32)

    def test_tf_output(self):
        output = random_mix_patch(tensor=self.test_tf, x=self.tf_x, y=self.tf_y, lam=self.tf_lam)
        with self.subTest('Check length of output tuple'):
            self.assertEqual(len(output), 6)
        with self.subTest('Check output value'):
            self.assertEqual(output, self.tf_output)

    def test_torch_output(self):
        output = random_mix_patch(tensor=self.test_torch, x=self.torch_x, y=self.torch_y, lam=self.torch_lam)
        with self.subTest('Check length of output tuple'):
            self.assertEqual(len(output), 6)
        with self.subTest('Check output value'):
            self.assertEqual(output, self.torch_output)
