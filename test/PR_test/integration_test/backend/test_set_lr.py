# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import tensorflow_addons as tfa
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestSetLr(unittest.TestCase):
    def test_set_lr_tf(self):
        m = fe.build(fe.architecture.tensorflow.LeNet, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
        fe.backend.set_lr(m, 2e-4)
        self.assertTrue(np.allclose(fe.backend.get_lr(model=m), 2e-4))

    def test_set_lr_tf_weight_decay(self):
        m = fe.build(fe.architecture.tensorflow.LeNet,
                     optimizer_fn=lambda: tfa.optimizers.SGDW(weight_decay=1e-5, learning_rate=1e-4))
        fe.backend.set_lr(m, 2e-4)
        self.assertTrue(np.allclose(tf.keras.backend.get_value(m.current_optimizer.weight_decay), 2e-5))

    def test_set_lr_torch(self):
        m = fe.build(fe.architecture.pytorch.LeNet, optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=1e-4))
        fe.backend.set_lr(m, 2e-4)
        self.assertTrue(np.allclose(fe.backend.get_lr(model=m), 2e-4))
