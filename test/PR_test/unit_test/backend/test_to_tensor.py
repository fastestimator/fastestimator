import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestToTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = {
            "x": np.ones((10, 15), dtype=np.int32),
            "y": [np.ones((4), dtype=np.float32), np.ones((5, 3), dtype=np.float32)],
            "z": {
                "key": np.ones((2, 2), dtype=np.float64)
            }
        }

        cls.data_tf = {
            "x": tf.ones((10, 15), dtype=tf.int32),
            "y": [tf.ones((4), dtype=tf.float32), tf.ones((5, 3), dtype=tf.float32)],
            "z": {
                "key": tf.ones((2, 2), dtype=tf.float64)
            }
        }
        cls.data_torch = {
            "x": torch.ones((10, 15), dtype=torch.int32),
            "y": [torch.ones((4), dtype=torch.float32), torch.ones((5, 3), dtype=torch.float32)],
            "z": {
                "key": torch.ones((2, 2), dtype=torch.float64)
            }
        }

    def test_to_tensor_np_to_tf(self):
        self.assertTrue(
            fet.is_equal(fe.backend.to_tensor(self.data_np, target_type='tf'), self.data_tf, assert_dtype=True))

    def test_to_tensor_np_to_torch(self):
        self.assertTrue(
            fet.is_equal(fe.backend.to_tensor(self.data_np, target_type='torch'), self.data_torch, assert_dtype=True))

    def test_to_tensor_tf_to_torch(self):
        self.assertTrue(
            fet.is_equal(fe.backend.to_tensor(self.data_tf, target_type='torch'), self.data_torch, assert_dtype=True))

    def test_to_tensor_torch_to_tf(self):
        self.assertTrue(
            fet.is_equal(fe.backend.to_tensor(self.data_torch, target_type='tf'), self.data_tf, assert_dtype=True))
