import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestToShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = {"x": np.ones((10, 15)), "y": [np.ones((4)), np.ones((5, 3))], "z": {"key": np.ones((2, 2))}}
        cls.data_tf = {"x": tf.ones((10, 15)), "y": [tf.ones((4)), tf.ones((5, 3))], "z": {"key": tf.ones((2, 2))}}
        cls.data_torch = {
            "x": torch.ones((10, 15)), "y": [torch.ones((4)), torch.ones((5, 3))], "z": {
                "key": torch.ones((2, 2))
            }
        }
        cls.op = {"x": (10, 15), "y": [(4, ), (5, 3)], "z": {"key": (2, 2)}}

        # shape = fe.backend.to_shape(data, add_batch=True)
        # {"x": (None, 10, 15), "y":[(None, 4), (None, 5, 3)], "z": {"key": (None, 2, 2)}}
        # shape = fe.backend.to_shape(data, exact_shape=False)

    def test_to_shape_np(self):
        self.assertEqual(fe.backend.to_shape(self.data_np), self.op)

    def test_to_shape_tf(self):
        self.assertEqual(fe.backend.to_shape(self.data_tf), self.op)

    def test_to_shape_torch(self):
        self.assertEqual(fe.backend.to_shape(self.data_torch), self.op)
