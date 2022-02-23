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
import random
import unittest

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

import fastestimator as fe
from fastestimator.trace.metric.bleu_score import BleuScore


def inputs():
    while True:
        ground_truth = random.sample(range(1, 100), 10)
        prediction = [i for i in ground_truth if random.random() > 0.3]
        prediction = prediction + [0] * (10 - len(prediction))
        yield {'target_real': ground_truth, 'pred': prediction}


def get_data():
    ground_truth = []
    pred = []
    for _ in range(100):
        data = inputs().__next__()
        ground_truth.append(data['target_real'])
        pred.append(data['pred'])
    return {'target_real': np.array(ground_truth), 'pred': np.array(pred)}


def get_expanded_input(data):
    return np.expand_dims(data, axis=1)


def get_formatted_list(data):
    filtered_data = []
    for i in data:
        filtered_data.append([j for j in i if j != 0])
    return filtered_data


class TestBleuScore(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.bleu_score = BleuScore(true_key="target_real",
                                    pred_key="pred",
                                    output_name="bleu_score",
                                    n_gram=4,
                                    mode="!train",
                                    per_ds=False)
        self.train_data = get_data()
        print(self.train_data['target_real'].shape)
        print(self.train_data['pred'].shape)

    def get_batch_data(self, batch_size):
        batch_data = []
        for i in range(0, 100, batch_size):
            gt = self.train_data['target_real'][i:i + batch_size]
            pred = self.train_data['pred'][i:i + batch_size]
            batch_data.append(fe.util.Data({'target_real': np.array(gt), 'pred': np.array(pred)}))
        return batch_data

    def test_weights(self):
        weights = self.bleu_score.weights
        self.assertEqual(weights, (0.25, 0.25, 0.25, 0.25))

    def test_invalid_ngram(self):
        with self.assertRaises(ValueError):
            _ = BleuScore(true_key="target_real", pred_key="pred", output_name="bleu_score", n_gram=-1, mode="!train")

    def test_corpus_bleu_score(self):

        batch_size = 4
        for batch_data in self.get_batch_data(batch_size):
            self.bleu_score.on_batch_end(batch_data)

        predicited_bleu_score = self.bleu_score.get_corpus_bleu_score()
        expected_bleu_score = corpus_bleu(get_expanded_input(self.train_data['target_real']),
                                          get_formatted_list(self.train_data['pred']),
                                          self.bleu_score.weights,
                                          smoothing_function=SmoothingFunction().method4)
        self.assertAlmostEqual(predicited_bleu_score, expected_bleu_score, delta=0.001)
