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

import fastestimator as fe
import numpy as np
import tensorflow as tf
import torch


class TestZscore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = np.array([[0, 1], [2, 3]])
        cls.data_tf = tf.constant([[0, 1], [2, 3]])
        cls.data_torch = torch.tensor([[0, 1], [2, 3]])
        cls.op = np.array([[-1.34164079, -0.4472136], [0.4472136, 1.34164079]])

    def test_zscore_np(self):
        self.assertTrue(np.allclose(fe.backend.zscore(self.data_np), self.op))

    def test_zscore_tf(self):
        self.assertTrue(np.allclose(fe.backend.zscore(self.data_tf).numpy(), self.op))

    def test_zscore_torch(self):
        self.assertTrue(np.allclose(fe.backend.zscore(self.data_torch).numpy(), self.op))
