import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import check_nan
from fastestimator.test.unittest_util import is_equal


class TestCheckNaN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np_nan = np.array([[[1.0, 2.0], [3.0, np.NaN]], [[5.0, 6.0], [7.0, 8.0]]])
        cls.data_np_inf = np.array([[[1.0, 2.0], [3.0, np.Inf]], [[5.0, 6.0], [7.0, 8.0]]])
        cls.data_tf_nan = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[np.NaN, 6.0], [7.0, 8.0]]])
        cls.data_tf_inf = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[np.Inf, 6.0], [7.0, 8.0]]])
        cls.data_torch_nan = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [np.NaN, 8.0]]])
        cls.data_torch_inf = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [np.Inf, 8.0]]])
        cls.op_np = np.bool_(True)
        cls.op_tf = tf.constant(True)
        cls.op_torch = torch.tensor(True)

    def test_check_nan_np_value(self):
        with self.subTest('Detect NaN values'):
            self.assertTrue(is_equal(check_nan(self.data_np_nan), self.op_np))
        with self.subTest('Detect Inf values'):
            self.assertTrue(is_equal(check_nan(self.data_np_inf), self.op_np))

    def test_check_nan_tf_value(self):
        with self.subTest('Detect NaN values'):
            self.assertTrue(is_equal(check_nan(self.data_tf_nan), self.op_tf))
        with self.subTest('Detect Inf values'):
            self.assertTrue(is_equal(check_nan(self.data_tf_inf), self.op_tf))

    def test_check_nan_torch_value(self):
        with self.subTest('Detect NaN values'):
            self.assertTrue(is_equal(check_nan(self.data_torch_nan), self.op_torch))
        with self.subTest('Detect Inf values'):
            self.assertTrue(is_equal(check_nan(self.data_torch_inf), self.op_torch))
