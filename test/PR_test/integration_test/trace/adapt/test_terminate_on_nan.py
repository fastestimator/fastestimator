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
from io import StringIO
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.test.unittest_util import one_layer_tf_model, sample_system_object
from fastestimator.trace.adapt import TerminateOnNaN
from fastestimator.trace.metric import Accuracy, F1Score
from fastestimator.util.data import Data


class TestTerminateOnNaN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = Data({'loss': np.NaN})
        cls.data_tf = Data({'loss': tf.constant(np.NaN)})
        cls.data_torch = Data({'loss': torch.tensor(np.NaN)})
        cls.expected_msg = "FastEstimator-TerminateOnNaN: NaN Detected in: loss"
        cls.expected_loss_keys = {"ce"}
        cls.expected_all_keys = {"ce", "accuracy", "f1_score"}

        tf_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        cls.network = fe.Network(ops=[
            ModelOp(model=tf_model, inputs="x", outputs="y"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=tf_model, loss_name="ce")
        ])
        cls.traces = [
            Accuracy(true_key="y", pred_key="y_pred", output_name="accuracy"),
            F1Score(true_key="y", pred_key="y_pred", output_name="f1_score")
        ]

    def test_on_epoch_begin_loss_keys(self):
        terminate_on_nan = TerminateOnNaN(monitor_names=None)
        terminate_on_nan.system = sample_system_object()
        terminate_on_nan.system.network = self.network
        terminate_on_nan.on_epoch_begin(data={})
        self.assertEqual(terminate_on_nan.monitor_keys, self.expected_loss_keys)

    def test_on_epoch_begin_all_keys(self):
        terminate_on_nan = TerminateOnNaN(monitor_names="*")
        terminate_on_nan.system = sample_system_object()
        terminate_on_nan.system.mode = "eval"
        terminate_on_nan.system.network = self.network
        terminate_on_nan.system.traces = self.traces
        terminate_on_nan.on_epoch_begin(data={})
        self.assertEqual(terminate_on_nan.monitor_keys, self.expected_all_keys)

    def test_on_batch_end(self):
        terminate_on_nan = TerminateOnNaN(monitor_names="loss")
        terminate_on_nan.system = sample_system_object()
        terminate_on_nan.on_epoch_begin(data={})
        with self.subTest('Numpy Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_batch_end(data=self.data_np)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)

        with self.subTest('Tf Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_batch_end(data=self.data_tf)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)

        with self.subTest('Torch Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_batch_end(data=self.data_torch)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)

    def test_on_epoch_end(self):
        terminate_on_nan = TerminateOnNaN(monitor_names="loss")
        terminate_on_nan.system = sample_system_object()
        terminate_on_nan.on_epoch_begin(data={})
        with self.subTest('Numpy Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_epoch_end(data=self.data_np)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)

        with self.subTest('Tf Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_epoch_end(data=self.data_tf)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)

        with self.subTest('Torch Data'):
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                terminate_on_nan.on_epoch_end(data=self.data_torch)
                log = fake_stdout.getvalue().strip()
                self.assertEqual(log, self.expected_msg)
