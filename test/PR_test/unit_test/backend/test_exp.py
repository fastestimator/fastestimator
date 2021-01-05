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


class TestExp(unittest.TestCase):
    def test_exp_np_input(self):
        n = np.array([-2., 2, 1])
        obj1 = fe.backend.exp(n)
        obj2 = np.array([0.13533528, 7.3890561, 2.71828183])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_exp_tf_input(self):
        t = tf.constant([-2., 2, 1])
        obj1 = fe.backend.exp(t)
        obj2 = np.array([0.13533528, 7.3890561, 2.71828183])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_exp_torch_input(self):
        t = torch.tensor([-2., 2, 1])
        obj1 = fe.backend.exp(t)
        obj2 = np.array([0.13533528, 7.3890561, 2.71828183])
        self.assertTrue(np.allclose(obj1, obj2))
