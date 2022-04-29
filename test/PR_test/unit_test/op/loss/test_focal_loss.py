# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

import tensorflow as tf
import torch

from fastestimator.backend import binary_crossentropy
from fastestimator.op.tensorop.loss.focal_loss import focal_loss


class TestFocalLoss(unittest.TestCase):
    def test_focal_loss_bc_tf(self):
        true = tf.constant([[1], [1], [1], [0], [0], [0]])
        pred = tf.constant([[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]])
        fl = focal_loss(y_pred=pred, y_true=true,
                        gamma=None, alpha=None)  # 0.1464
        bc = binary_crossentropy(y_pred=pred, y_true=true)
        self.assertAlmostEqual(bc, fl, delta=0.0001)

    def test_focal_loss_tf(self):
        true = tf.constant([[1], [1], [1], [0], [0], [0]])
        pred = tf.constant([[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]])
        fl = focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25)
        self.assertAlmostEqual(0.004, fl, delta=0.0001)

    def test_focal_loss_bc_torch(self):
        true = torch.tensor([[1], [1], [1], [0], [0], [0]]).to(torch.float32)
        pred = torch.tensor([[0.97], [0.91], [0.73], [0.27], [
                            0.09], [0.03]]).to(torch.float32)
        fl = focal_loss(y_pred=pred, y_true=true, gamma=None, alpha=None)
        bc = binary_crossentropy(y_pred=pred, y_true=true)
        self.assertAlmostEqual(bc, fl, delta=0.0001)

    def test_focal_loss_torch(self):
        true = torch.tensor([[1], [1], [1], [0], [0], [0]]).to(torch.float32)
        pred = torch.tensor([[0.97], [0.91], [0.73], [0.27], [
                            0.09], [0.03]]).to(torch.float32)
        fl = focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25)
        self.assertAlmostEqual(0.004, fl, delta=0.0001)
