# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

    def test_to_shape_np(self):
        self.assertEqual(fe.backend.to_shape(self.data_np), self.op)

    def test_to_shape_tf(self):
        self.assertEqual(fe.backend.to_shape(self.data_tf), self.op)

    def test_to_shape_torch(self):
        self.assertEqual(fe.backend.to_shape(self.data_torch), self.op)
