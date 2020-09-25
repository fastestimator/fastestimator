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
from fastestimator.test.unittest_util import is_equal


class TestTranspose(unittest.TestCase):
    def test_np(self):
        n = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = fe.backend.transpose(n)
        self.assertTrue(is_equal(b, np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])))

    def test_tf(self):
        t = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = fe.backend.transpose(t)
        self.assertTrue(is_equal(b, tf.constant([[0, 3, 6], [1, 4, 7], [2, 5, 8]])))

    def test_torch(self):
        p = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        b = fe.backend.transpose(p)
        self.assertTrue(is_equal(b, torch.tensor([[0, 3, 6], [1, 4, 7], [2, 5, 8]])))
