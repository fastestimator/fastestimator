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


class TestAbs(unittest.TestCase):
    def test_abs_np_input(self):
        n = np.array([-2, 7, -19])
        obj1 = fe.backend.abs(n)
        obj2 = np.array([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))

    def test_abs_tf_input(self):
        t = tf.constant([-2, 7, -19])
        obj1 = fe.backend.abs(t)
        obj2 = tf.constant([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))

    def test_abs_torch_input(self):
        t = torch.tensor([-2, 7, -19])
        obj1 = fe.backend.abs(t)
        obj2 = torch.tensor([2, 7, 19])
        self.assertTrue(is_equal(obj1, obj2))
