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
import math
import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend import dice_score


class TestDiceScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        true = np.zeros((5, 20, 20, 3), dtype=np.float32)
        pred = np.zeros((5, 20, 20, 3), dtype=np.float32)

        # Slice 0 is a blank
        # Slice 1 is a total false negative
        true[1, 5:10, 5:10, 0] = 1.0
        true[1, 7:10, :, 1] = 1.0
        true[1, 2:4, 3:6, 2] = 1.0
        # Slice 2 is a total false positive
        pred[2, 5:10, 5:10, 0] = 1.0
        pred[2, 7:10, :, 1] = 1.0
        pred[2, 2:4, 3:6, 2] = 1.0
        # Slice 3 is a partial match
        true[3, 5:10, 5:10, 0] = 1.0
        true[3, 7:10, :, 1] = 1.0
        true[3, 2:4, 3:6, 2] = 1.0
        pred[3, 3:10, 5:10, 0] = 0.8
        pred[3, 7:10, :, 1] = 0.7
        pred[3, 3:8, 3:6, 2] = 0.6
        # Slice 4 is a perfect match
        true[4, 5:10, 5:10, 0] = 1.0
        true[4, 7:10, :, 1] = 1.0
        true[4, 2:4, 3:6, 2] = 1.0
        pred[4, 5:10, 5:10, 0] = 1.0
        pred[4, 7:10, :, 1] = 1.0
        pred[4, 2:4, 3:6, 2] = 1.0

        cls.np_true = np.array(true)
        cls.np_pred = np.array(pred)
        cls.torch_true = torch.tensor(true).permute(0, 3, 1, 2)  # Torch is channel first
        cls.torch_pred = torch.tensor(pred).permute(0, 3, 1, 2)
        cls.tf_true = tf.constant(true)
        cls.tf_pred = tf.constant(pred)

    def test_dice_score(self):
        np_dice_score = dice_score(self.np_pred, self.np_true)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true)

        target = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.75472, 0.82353, 0.24],
                           [1.0, 1.0, 1.0]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_empty_nan(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, empty_nan=True)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, empty_nan=True)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, empty_nan=True)

        target = np.array([[math.nan, math.nan, math.nan],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.75472, 0.82353, 0.24],
                           [1.0, 1.0, 1.0]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_threshold_score(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, threshold=0.65)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, threshold=0.65)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, threshold=0.65)

        target = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.83333, 1.0, 0.0],
                           [1.0, 1.0, 1.0]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_exclusive_channels(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, mask_overlap=False)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, mask_overlap=False)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, mask_overlap=False)

        target = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.75472, 0.688852, 0.22222],
                           [1.0, 1.0, 1.0]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 3)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 3)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 3)

    def test_soft_dice_loss(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, soft_dice=True)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, soft_dice=True)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, soft_dice=True)

        target = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.84388, 0.93960, 0.31579],
                           [1.0, 1.0, 1.0]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_channel_weights(self):
        weights = np.array([[1.0, 2.0, 0.5]])
        np_dice_score = dice_score(self.np_pred, self.np_true, channel_weights=weights)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, channel_weights=tf.convert_to_tensor(weights))
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, channel_weights=torch.Tensor(weights))

        target = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.75472, 0.82353*2, 0.24*0.5],
                           [1.0, 2.0, 0.5]])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_channel_average(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, channel_average=True)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, channel_average=True)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, channel_average=True)

        target = np.array([0.0, 0.0, 0.0, 0.606083, 1.0])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_sample_average(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, sample_average=True)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, sample_average=True)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, sample_average=True)

        target = np.array([0.350944, 0.364706, 0.248])

        np.testing.assert_array_almost_equal(np_dice_score, target, 4)
        np.testing.assert_array_almost_equal(tf_dice_score, target, 4)
        np.testing.assert_array_almost_equal(torch_dice_score, target, 4)

    def test_full_average(self):
        np_dice_score = dice_score(self.np_pred, self.np_true, channel_average=True, sample_average=True)
        tf_dice_score = dice_score(self.tf_pred, self.tf_true, channel_average=True, sample_average=True)
        torch_dice_score = dice_score(self.torch_pred, self.torch_true, channel_average=True, sample_average=True)

        self.assertAlmostEqual(np_dice_score, 0.3212166, places=4)
        self.assertAlmostEqual(tf_dice_score.numpy(), 0.3212166, places=4)
        self.assertAlmostEqual(torch_dice_score.numpy(), 0.3212166, places=4)
