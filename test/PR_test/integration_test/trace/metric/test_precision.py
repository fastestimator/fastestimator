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
from fastestimator.trace.metric import Precision


class TestPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p_key = "precision"

    def test_passing_kwarg(self):
        with self.subTest("illegal kwargs"):
            with self.assertRaises(ValueError):
                # `average` is illegal karg
                trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key, average="binary")

        with self.subTest("check if kwargs pass to precision_score"):
            with unittest.mock.patch("fastestimator.trace.metric.precision.precision_score") as fake:
                kwargs = {"e1": "extra1", "e2": "extra2"}
                trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key, **kwargs)
                batch = {"label": tf.constant([0, 1, 0, 1])}
                pred = {"pred": tf.constant([[0.2], [0.6], [0.8], [0.1]])}  # [[0], [1], [1], [0]]
                run = TraceRun(trace=trace, batch=batch, prediction=pred)
                run.run_trace()

            fake_kwargs = fake.call_args[1]
            for key, val in kwargs.items():
                self.assertTrue(key in fake_kwargs)
                self.assertEqual(val, fake_kwargs[key])

    def test_tf_binary_class(self):
        with self.subTest("ordinal label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            # tp, tn, fp, fn = [1, 1, 1, 1]
            batch = {"label": tf.constant([0, 1, 0, 1])}
            pred = {"pred": tf.constant([[0.2], [0.6], [0.8], [0.1]])}  # [[0], [1], [1], [0]]
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key], 0.5)  # precision = tp / (tp + fp) = 0.5

        with self.subTest("one-hot label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            # tp, tn, fp, fn = [2, 1, 0, 1]
            batch = {"label": tf.constant([[1, 0], [0, 1], [0, 1], [0, 1]])}  #  [0, 1, 1, 1]
            pred = {"pred": tf.constant([[0.2], [0.6], [0.8], [0.1]])}  #  [[0], [1], [1], [0]]
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key], 1)  # precision = tp / (tp + fp) = 1

    def test_torch_binary_class(self):
        with self.subTest("ordinal label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            # tp, tn, fp, fn = [1, 1, 1, 1]
            batch = {"label": torch.tensor([0, 1, 0, 1])}
            pred = {"pred": torch.tensor([[0.2], [0.6], [0.8], [0.1]])}  # [[0], [1], [1], [0]]
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key], 0.5)  # precision = tp / (tp + fp) = 0.5

        with self.subTest("one-hot label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            # tp, tn, fp, fn = [2, 1, 0, 1]
            batch = {"label": torch.tensor([[1, 0], [0, 1], [0, 1], [0, 1]])}  #  [0, 1, 1, 1]
            pred = {"pred": torch.tensor([[0.2], [0.6], [0.8], [0.1]])}  #  [[0], [1], [1], [0]]
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key], 1.0)  # precision = tp / (tp + fp) = 1

    def test_tf_multi_class(self):
        with self.subTest("ordinal label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            batch = {"label": tf.constant([0, 0, 0, 1, 1, 2])}
            pred = {
                "pred":
                tf.constant([[0.2, 0.1, -0.6], [0.6, 2.0, 0.1], [0.1, 0.1, 0.8], [0.4, 0.1, -0.3], [0.2, 0.7, 0.1],
                             [0.3, 0.6, 1.5]])  # [[0], [1], [2], [0], [1], [2]]
            }
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key][0],
                             0.5)  # for 0, [tp, tn, fp, fn] = [1, 2, 1, 2], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][1],
                             0.5)  # for 1, [tp, tn, fp, fn] = [1, 3, 1, 1], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][2],
                             0.5)  # for 2, [tp, tn, fp, fn] = [1, 4, 1, 0], precision = 0.5

        with self.subTest("one-hot label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            batch = {
                "label": tf.constant([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
            }  # [0, 0, 0, 1, 1, 2]
            pred = {
                "pred":
                tf.constant([[0.2, 0.1, -0.6], [0.6, 2.0, 0.1], [0.1, 0.1, 0.8], [0.4, 0.1, -0.3], [0.2, 0.7, 0.1],
                             [0.3, 0.6, 1.5]])  # [[0], [1], [2], [0], [1], [2]]
            }
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key][0],
                             0.5)  # for 0, [tp, tn, fp, fn] = [1, 2, 1, 2], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][1],
                             0.5)  # for 1, [tp, tn, fp, fn] = [1, 3, 1, 1], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][2],
                             0.5)  # for 2, [tp, tn, fp, fn] = [1, 4, 1, 0], precision = 0.5

    def test_torch_multi_class(self):
        with self.subTest("ordinal label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            batch = {"label": torch.tensor([0, 0, 0, 1, 1, 2])}
            pred = {
                "pred":
                torch.tensor([[0.2, 0.1, -0.6], [0.6, 2.0, 0.1], [0.1, 0.1, 0.8], [0.4, 0.1, -0.3], [0.2, 0.7, 0.1],
                              [0.3, 0.6, 1.5]])  # [[0], [1], [2], [0], [1], [2]]
            }
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key][0],
                             0.5)  # for 0, [tp, tn, fp, fn] = [1, 2, 1, 2], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][1],
                             0.5)  # for 1, [tp, tn, fp, fn] = [1, 3, 1, 1], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][2],
                             0.5)  # for 2, [tp, tn, fp, fn] = [1, 4, 1, 0], precision = 0.5

        with self.subTest("one-hot label"):
            trace = Precision(true_key="label", pred_key="pred", output_name=self.p_key)
            batch = {
                "label": torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
            }  # [0, 0, 0, 1, 1, 2]
            pred = {
                "pred":
                torch.tensor([[0.2, 0.1, -0.6], [0.6, 2.0, 0.1], [0.1, 0.1, 0.8], [0.4, 0.1, -0.3], [0.2, 0.7, 0.1],
                              [0.3, 0.6, 1.5]])  # [[0], [1], [2], [0], [1], [2]]
            }
            run = TraceRun(trace=trace, batch=batch, prediction=pred)
            run.run_trace()
            self.assertEqual(run.data_on_epoch_end[self.p_key][0],
                             0.5)  # for 0, [tp, tn, fp, fn] = [1, 2, 1, 2], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][1],
                             0.5)  # for 1, [tp, tn, fp, fn] = [1, 3, 1, 1], precision = 0.5
            self.assertEqual(run.data_on_epoch_end[self.p_key][2],
                             0.5)  # for 2, [tp, tn, fp, fn] = [1, 4, 1, 0], precision = 0.5
