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

from fastestimator.backend.tensor_normalize import normalize


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_np = np.array([[1, 2], [3, 3]])
        self.test_tf = tf.cast(tf.constant([[1, 1], [2, -3], [4, 1]]), tf.float32)
        self.test_torch = torch.Tensor([[1, 1], [2, -3], [4, 1]])

    def test_normalize_np_value(self):
        np.testing.assert_array_almost_equal(normalize(self.test_np, None, None), np.array([[-1.50755654, -0.30151131], [ 0.90453392,  0.90453392]]))

    def test_normalize_tf_value(self):
        np.testing.assert_array_almost_equal(normalize(self.test_tf, None, None).numpy(), np.array([[ 0. ,  0.], [ 0.48038446, -1.92153785], [ 1.44115338,  0.]]))

    def test_normalize_torch_value(self):
        np.testing.assert_array_almost_equal(normalize(self.test_torch, None, None).numpy(), np.array([[ 0. ,  0.], [ 0.48038446, -1.92153785], [ 1.44115338,  0.]]))

