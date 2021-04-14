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

import tensorflow as tf
import torch

from fastestimator.test.unittest_util import TraceRun
from fastestimator.trace.metric import Accuracy


class TestAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc_key = "acc"

    def test_tf_one_hot_label(self):
        trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key)
        batch = {"label": tf.constant([[1, 0, 0], [0, 1, 0]])}  # one-hot
        prediction = {"pred": tf.constant([[1, 2, 3], [0.2, 0.5, 0.3]])}
        sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
        sim.run_trace()
        self.assertEqual(sim.data_on_epoch_end[self.acc_key], 0.5)

    def test_tf_class_index_label(self):
        trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key)
        batch = {"label": tf.constant([0, 1])}  # class index
        prediction = {"pred": tf.constant([[1, 2, 3], [0.2, 0.5, 0.3]])}
        sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
        sim.run_trace()
        self.assertEqual(sim.data_on_epoch_end[self.acc_key], 0.5)

    def test_tf_binary_class(self):
        with self.subTest("from_logit=False"):
            trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key, from_logits=False)
            batch = {"label": tf.constant([0, 1])}
            prediction = {"pred": tf.constant([[0.3], [0.6]])}  # pred > 0.5 => class 1
            sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
            sim.run_trace()
            self.assertEqual(sim.data_on_epoch_end[self.acc_key], 1.0)

        with self.subTest("from_logit=True"):
            trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key, from_logits=True)
            batch = {"label": tf.constant([0, 1])}
            prediction = {"pred": tf.constant([[-1], [1]])}  # 1 / 1 + exp(-pred) > 0.5 => class 1
            sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
            sim.run_trace()
            self.assertEqual(sim.data_on_epoch_end[self.acc_key], 1.0)

    def test_torch_one_hot_label(self):
        trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key)
        batch = {"label": torch.tensor([[1, 0, 0], [0, 1, 0]])}  # one-hot
        prediction = {"pred": torch.tensor([[1, 2, 3], [0.2, 0.5, 0.3]])}
        sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
        sim.run_trace()
        self.assertEqual(sim.data_on_epoch_end[self.acc_key], 0.5)

    def test_torch_class_index_label(self):
        trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key)
        batch = {"label": torch.tensor([0, 1])}  # class index
        prediction = {"pred": torch.tensor([[1, 2, 3], [0.2, 0.5, 0.3]])}
        sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
        sim.run_trace()
        self.assertEqual(sim.data_on_epoch_end[self.acc_key], 0.5)

    def test_torch_binary_class(self):
        with self.subTest("from_logit=False"):
            trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key, from_logits=False)
            batch = {"label": torch.tensor([0, 1])}
            prediction = {"pred": torch.tensor([[0.3], [0.6]])}  # pred > 0.5 => class 1
            sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
            sim.run_trace()
            self.assertEqual(sim.data_on_epoch_end[self.acc_key], 1.0)

        with self.subTest("from_logit=True"):
            trace = Accuracy(true_key="label", pred_key="pred", output_name=self.acc_key, from_logits=True)
            batch = {"label": torch.tensor([0, 1])}
            prediction = {"pred": torch.tensor([[-1], [1]])}  # 1 / 1 + exp(-pred) > 0.5 => class 1
            sim = TraceRun(trace=trace, batch=batch, prediction=prediction)
            sim.run_trace()
            self.assertEqual(sim.data_on_epoch_end[self.acc_key], 1.0)
