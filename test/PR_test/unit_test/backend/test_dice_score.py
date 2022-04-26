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

from fastestimator.backend import dice_score


class TestDiceScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.true = [[[[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                     [[0, 1, 1], [1, 0, 1], [1, 0, 1]],
                     [[0, 1, 1], [1, 0, 1], [1, 0, 1]]]]

        cls.pred = [[[[0, 1, 0], [1, 0, 0], [1, 0, 1]],
                     [[0, 1, 1], [1, 0, 1], [0, 0, 0]],
                     [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]]
        cls.np_true = np.array(cls.true, dtype=np.float32)
        cls.np_pred = np.array(cls.pred, dtype=np.float32)
        cls.torch_true = torch.tensor(cls.true).type(torch.float32)
        cls.torch_pred = torch.tensor(cls.pred).type(torch.float32)
        cls.tf_true = tf.constant(cls.true, dtype=tf.float32)
        cls.tf_pred = tf.constant(cls.pred, dtype=tf.float32)

    def test_dice_score(cls):
        np_dice_score = dice_score(cls.np_true, cls.np_pred)
        tf_dice_score = dice_score(cls.tf_true, cls.tf_pred)
        torch_dice_score = dice_score(cls.torch_true, cls.torch_pred)
        cls.assertAlmostEqual(np_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(tf_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(torch_dice_score, 0.839, delta=0.001)

    def test_soft_dice_loss(cls):
        np_dice_score = dice_score(
            cls.np_true, cls.np_pred, soft_dice=True)
        tf_dice_score = dice_score(
            cls.tf_true, cls.tf_pred, soft_dice=True)
        torch_dice_score = dice_score(
            cls.torch_true, cls.torch_pred, soft_dice=True)
        cls.assertAlmostEqual(np_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(tf_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(torch_dice_score, 0.839, delta=0.001)

    def test_channel_average(cls):
        np_dice_score = dice_score(
            cls.np_true, cls.np_pred, channel_average=True)
        tf_dice_score = dice_score(
            cls.tf_true, cls.tf_pred, channel_average=True)
        torch_dice_score = dice_score(
            cls.torch_true, cls.torch_pred, channel_average=True)
        cls.assertAlmostEqual(np_dice_score, 0.837, delta=0.001)
        cls.assertAlmostEqual(tf_dice_score, 0.837, delta=0.001)
        cls.assertAlmostEqual(torch_dice_score, 0.837, delta=0.001)

    def test_average_sample_loss(cls):
        np_dice_score = dice_score(
            cls.np_true, cls.np_pred, sample_average=True)
        tf_dice_score = dice_score(
            cls.tf_true, cls.tf_pred, sample_average=True)
        torch_dice_score = dice_score(
            cls.torch_true, cls.torch_pred, sample_average=True)
        cls.assertAlmostEqual(np_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(tf_dice_score, 0.839, delta=0.001)
        cls.assertAlmostEqual(torch_dice_score, 0.839, delta=0.001)

    def test_dice_loss_2d(cls):
        np_true = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=np.float32)
        np_pred = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float32)
        np_dice_score = dice_score(np_true, np_pred)
        cls.assertAlmostEqual(np_dice_score, 0.7692, delta=0.001)
