# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.backend import reduce_std


class TestReduceStd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_np = np.array([[1, 2], [3, 3]])
        self.test_tf = tf.cast(tf.constant([[1, 1], [2, -3], [4, 2]]), tf.float32)
        self.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 2]])

    def test_reduce_mean_np_type(self):
        self.assertIsInstance(reduce_std(self.test_np), np.ScalarType, 'Output type must be NumPy')

    def test_reduce_mean_np_value(self):
        self.assertAlmostEqual(reduce_std(self.test_np), 0.829, delta=0.001)

    def test_reduce_mean_tf_type(self):
        self.assertIsInstance(reduce_std(self.test_tf), tf.Tensor, 'Output type must be tf.Tensor')

    def test_reduce_mean_tf_value(self):
        self.assertAlmostEqual(reduce_std(self.test_tf).numpy(), 2.114, delta=0.001)

    def test_reduce_mean_torch_type(self):
        self.assertIsInstance(reduce_std(self.test_torch), torch.Tensor, 'Output type must be torch.Tensor')

    def test_reduce_mean_torch_value(self):
        self.assertAlmostEqual(reduce_std(self.test_torch).numpy(), 2.114, delta=0.001)
