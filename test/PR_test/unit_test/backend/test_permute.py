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


class TestPermute(unittest.TestCase):
    def test_permute_np(self):
        n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        with self.subTest("permutation case 1"):
            obj1 = fe.backend.permute(n, [2, 0, 1])
            obj2 = np.array([[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("permutation case 2"):
            obj1 = fe.backend.permute(n, [0, 2, 1])
            obj2 = np.array([[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_permute_tf(self):
        t = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        with self.subTest("permutation case 1"):
            obj1 = fe.backend.permute(t, [2, 0, 1])
            obj2 = tf.constant([[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("permutation case 2"):
            obj1 = fe.backend.permute(t, [0, 2, 1])
            obj2 = tf.constant([[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))

    def test_permute_torch(self):
        t = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        with self.subTest("permutation case 1"):
            obj1 = fe.backend.permute(t, [2, 0, 1])
            obj2 = torch.tensor([[[0, 2], [4, 6], [8, 10]], [[1, 3], [5, 7], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))

        with self.subTest("permutation case 2"):
            obj1 = fe.backend.permute(t, [0, 2, 1])
            obj2 = torch.tensor([[[0, 2], [1, 3]], [[4, 6], [5, 7]], [[8, 10], [9, 11]]])
            self.assertTrue(is_equal(obj1, obj2))
