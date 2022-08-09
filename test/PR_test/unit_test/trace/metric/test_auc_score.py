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
from sklearn.metrics import roc_auc_score

from fastestimator.trace.metric.auc import AUCScore


class TestAUCScore(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.auc_score = AUCScore(true_key="target_real",
                                  pred_key="pred",
                                  output_name="auc",
                                  mode="!train",
                                  per_ds=False)
        ground_truth = np.array([0, 0, 1, 1])
        prediction = np.array([0.1, 0.4, 0.35, 0.8])
        self.train_data = {'target_real': ground_truth, 'pred': prediction}
        self.auc_score.on_epoch_begin(None)

    def test_auc_score_binary(self):
        auc_score = AUCScore(true_key="target_real", pred_key="pred", output_name="auc", mode="!train", per_ds=False)
        auc_score.on_epoch_begin(None)
        ground_truth = np.array([0, 0, 1, 1])
        prediction = np.array([0.1, 0.4, 0.35, 0.8])
        train_data = {'target_real': ground_truth, 'pred': prediction}
        self.auc_score.on_batch_end(train_data)
        predicited_auc_score = self.auc_score.get_auc()
        expected_auc_score = roc_auc_score(ground_truth, prediction)
        self.assertAlmostEqual(predicited_auc_score, expected_auc_score, delta=0.001)

    def test_auc_score_multiclass_ovr(self):

        auc_score = AUCScore(true_key="target_real",
                             pred_key="pred",
                             output_name="auc",
                             multi_class='ovr',
                             mode="!train",
                             per_ds=False)
        auc_score.on_epoch_begin(None)
        ground_truth = np.array([0, 0, 1, 1, 2, 2, 2])
        prediction = np.array([[0.5, 0.4, 0.1], [0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.2, 0.8, 0.0], [0.2, 0.1, 0.7],
                               [0.1, 0.3, 0.6], [0.2, 0.1, 0.7]])
        train_data = {'target_real': ground_truth, 'pred': prediction}
        auc_score.on_batch_end(train_data)
        predicited_auc_score = auc_score.get_auc()
        expected_auc_score = roc_auc_score(
            ground_truth,
            prediction,
            multi_class='ovr',
        )
        self.assertAlmostEqual(predicited_auc_score, expected_auc_score, delta=0.001)

    def test_auc_score_multiclass_ovo(self):

        auc_score = AUCScore(true_key="target_real",
                             pred_key="pred",
                             output_name="auc",
                             multi_class='ovo',
                             mode="!train",
                             per_ds=False)
        auc_score.on_epoch_begin(None)
        ground_truth = np.array([0, 0, 1, 1, 2, 2, 2])
        prediction = np.array([[0.5, 0.4, 0.1], [0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.2, 0.8, 0.0], [0.2, 0.1, 0.7],
                               [0.1, 0.3, 0.6], [0.2, 0.1, 0.7]])
        train_data = {'target_real': ground_truth, 'pred': prediction}
        auc_score.on_batch_end(train_data)
        predicited_auc_score = auc_score.get_auc()
        expected_auc_score = roc_auc_score(
            ground_truth,
            prediction,
            multi_class='ovo',
        )
        self.assertAlmostEqual(predicited_auc_score, expected_auc_score, delta=0.001)

    def test_auc_score_multiclass_error(self):
        with self.assertRaises(ValueError):
            auc_score = AUCScore(true_key="target_real",
                                 pred_key="pred",
                                 output_name="auc",
                                 mode="!train",
                                 per_ds=False)
            auc_score.on_epoch_begin(None)
            ground_truth = np.array([0, 0, 1, 1, 2, 2, 2])
            prediction = np.array([[0.5, 0.4, 0.1], [0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.2, 0.8, 0.0], [0.2, 0.1, 0.7],
                                   [0.1, 0.3, 0.6], [0.2, 0.1, 0.7]])
            train_data = {'target_real': ground_truth, 'pred': prediction}
            auc_score.on_batch_end(train_data)
            _ = auc_score.get_auc()
