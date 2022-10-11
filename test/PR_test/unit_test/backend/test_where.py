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

from fastestimator.backend import where


class TestWhere(unittest.TestCase):

    def test_np(self):
        n = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = where(n > 4, n, -1)
        target = np.array([[-1, -1, -1], [-1, -1, 5], [6, 7, 8]])
        self.assertTrue(np.array_equal(b, target))

    def test_tf(self):
        n = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = where(n > 4, n, -1)
        target = np.array([[-1, -1, -1], [-1, -1, 5], [6, 7, 8]])
        self.assertTrue(np.array_equal(b.numpy(), target))

    def test_torch(self):
        n = torch.Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = where(n > 4, n, -1)
        target = np.array([[-1, -1, -1], [-1, -1, 5], [6, 7, 8]])
        self.assertTrue(np.array_equal(b.numpy(), target))
