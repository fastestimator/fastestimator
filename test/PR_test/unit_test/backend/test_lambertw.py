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
import math
import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe


class TestLambertW(unittest.TestCase):
    def test_lambertw_np_input(self):
        n = np.array([-1.0 / math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
        obj1 = fe.backend.lambertw(n)
        obj2 = np.array([-1.0, -0.653695, -0.560489, -0.259171, 0, 0.107743, 0.131515, 1, 1.32672, math.e, 3.38563])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_lambertw_tf_input(self):
        t = tf.constant([-1.0 / math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
        obj1 = fe.backend.lambertw(t)
        obj2 = np.array([-1.0, -0.653695, -0.560489, -0.259171, 0, 0.107743, 0.131515, 1, 1.32672, math.e, 3.38563])
        self.assertTrue(np.allclose(obj1, obj2, atol=1e-3))

    def test_lambertw_torch_input(self):
        t = torch.tensor([-1.0 / math.e, -0.34, -0.32, -0.2, 0, 0.12, 0.15, math.e, 5, math.exp(1 + math.e), 100])
        obj1 = fe.backend.lambertw(t)
        obj2 = np.array([-1.0, -0.653695, -0.560489, -0.259171, 0, 0.107743, 0.131515, 1, 1.32672, math.e, 3.38563])
        self.assertTrue(np.allclose(obj1, obj2, atol=1e-6))
