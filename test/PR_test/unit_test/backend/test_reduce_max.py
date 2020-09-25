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

from fastestimator.backend import reduce_max


class TestReduceMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.array([[1, 2], [3, 3]])
        cls.test_tf = tf.constant([[1, 1], [2, -3], [4, 1]])
        cls.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_reduce_max_np_type(self):
        self.assertIsInstance(reduce_max(self.test_np), np.ScalarType, 'Output type must be NumPy')

    def test_reduce_max_np_value(self):
        self.assertEqual(reduce_max(self.test_np), 3)

    def test_reduce_max_tf_type(self):
        self.assertIsInstance(reduce_max(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_reduce_max_tf_value(self):
        self.assertTrue(np.array_equal(reduce_max(self.test_tf).numpy(), 4))

    def test_reduce_max_torch_type(self):
        self.assertIsInstance(reduce_max(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_reduce_max_torch_value(self):
        self.assertTrue(np.array_equal(reduce_max(self.test_torch).numpy(), 4))

    def test_reduce_max_axis_value(self):
        self.assertTrue(np.array_equal(reduce_max(self.test_np, axis=0), [3, 3]))
