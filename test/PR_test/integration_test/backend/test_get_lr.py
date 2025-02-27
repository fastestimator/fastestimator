# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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


class TestGetLr(unittest.TestCase):

    def test_get_lr_tf(self):
        m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn=lambda: tf.keras.optimizers.legacy.Adam(1e-4))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 1e-4))

        m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn=lambda: tf.keras.optimizers.legacy.Adam(5e-2))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 5e-2))

    def test_get_lr_torch(self):
        m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1e-4))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 1e-4))

        m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=5e-2))
        b = fe.backend.get_lr(model=m)
        self.assertTrue(np.allclose(b, 5e-2))
