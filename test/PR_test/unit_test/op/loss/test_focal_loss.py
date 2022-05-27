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

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import binary_crossentropy
from fastestimator.op.tensorop.loss.focal_loss import focal_loss


class TestFocalLoss(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                             dtype=np.float32)
        self.pred = np.array([[0.6, 0.2, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1], [0.2, 0.2, 0.5, 0.1], [0.6, 0.1, 0.1, 0.2],
                              [0.4, 0.1, 0.2, 0.3], [0.0, 0.4, 0.6, 0.0]],
                             dtype=np.float32)

        self.true_tf_seg = np.array(
            [[[[1], [0], [0], [0]], [[0], [1], [0], [0]], [[0], [0], [1], [0]], [[1], [0], [0], [0]]],
             [[[1], [0], [0], [0]], [[0], [1], [0], [0]], [[1], [0], [0], [0]], [[0], [1], [0], [0]]]],
            dtype=np.float32)

        self.pred_tf_seg = np.array([[[[0.6], [0.2], [0.6], [0.1]], [[0.3], [0.9], [0.4], [0.2]],
                                      [[0.3], [0.2], [0.7], [0]], [[0.4], [0.1], [0.2], [0.4]]],
                                     [[[0.9], [0.3], [0.5], [0.3]], [[0.4], [0.8], [0.2], [0.3]],
                                      [[0.7], [0.1], [0.2], [0.5]], [[0.4], [0.6], [0.3], [0.4]]]],
                                    dtype=np.float32)

    def test_focal_loss_tf(self):
        fl = focal_loss(
            y_pred=tf.constant(self.pred),
            y_true=tf.constant(self.true),
            gamma=2.0,
            alpha=0.25,
        )
        self.assertAlmostEqual(0.112, fl, delta=0.01)

    def test_focal_loss_tf_3d(self):
        fl = focal_loss(y_pred=tf.constant(self.pred_tf_seg),
                        y_true=tf.constant(self.true_tf_seg),
                        gamma=2.0,
                        alpha=0.25)
        self.assertAlmostEqual(0.142, fl, delta=0.01)

    def test_focal_loss_tf_4d(self):
        true = tf.reshape(tf.constant(self.true_tf_seg), (2, 1, 4, 4, 1))
        pred = tf.reshape(tf.constant(self.pred_tf_seg), (2, 1, 4, 4, 1))
        fl = focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25)
        self.assertAlmostEqual(0.142, fl, delta=0.01)

    def test_focal_loss_torch(self):
        fl = focal_loss(y_pred=torch.tensor(self.pred), y_true=torch.tensor(self.true), gamma=2.0, alpha=0.25)
        self.assertAlmostEqual(0.112, fl, delta=0.01)

    def test_focal_loss_torch_3d(self):
        fl = focal_loss(y_pred=torch.tensor(self.pred_tf_seg),
                        y_true=torch.tensor(self.true_tf_seg),
                        gamma=2.0,
                        alpha=0.25)
        self.assertAlmostEqual(0.142, fl, delta=0.01)

    def test_focal_loss_torch_4d(self):
        true = torch.tensor(self.true_tf_seg).view((2, 1, 4, 4, 1))
        pred = torch.tensor(self.pred_tf_seg).view((2, 1, 4, 4, 1))
        fl = focal_loss(y_pred=pred, y_true=true, gamma=2.0, alpha=0.25)
        self.assertAlmostEqual(0.142, fl, delta=0.01)
