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
import math
import unittest
from io import StringIO
from unittest.mock import patch

import fastestimator as fe
from fastestimator.test.unittest_util import MultiLayerTorchModel, one_layer_tf_model, sample_system_object
from fastestimator.trace.adapt import ReduceLROnPlateau
from fastestimator.util.data import Data


class TestReduceLROnPlateau(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})
        cls.tf_expected_msg = "FastEstimator-ReduceLROnPlateau: learning rate reduced to 0.00010000000474974513"
        cls.torch_expected_msg = "FastEstimator-ReduceLROnPlateau: learning rate reduced to 9.999999747378752e-05"

    def test_tf_model_on_epoch_end_reduce_lr(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        model_name = model.model_name + '_lr'
        lr_on_plateau = ReduceLROnPlateau(model=model, metric='loss')
        lr_on_plateau.system = sample_system_object()
        lr_on_plateau.best = 5
        lr_on_plateau.wait = 11
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            lr_on_plateau.on_epoch_end(data=self.data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, self.tf_expected_msg)
        with self.subTest('Check learning rate in data'):
            self.assertTrue(math.isclose(self.data[model_name], 0.000100000005, rel_tol=1e-3))

    def test_tf_model_on_epoch_end_lr_wait(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        lr_on_plateau = ReduceLROnPlateau(model=model, metric='loss')
        lr_on_plateau.system = sample_system_object()
        lr_on_plateau.best = 12
        lr_on_plateau.on_epoch_end(data=self.data)
        with self.subTest('Check value of wait'):
            self.assertEqual(lr_on_plateau.wait, 0)
        with self.subTest('Check value of best'):
            self.assertEqual(lr_on_plateau.best, 10)

    def test_torch_model_on_epoch_end_reduce_lr(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        model_name = model.model_name + '_lr'
        lr_on_plateau = ReduceLROnPlateau(model=model, metric='loss')
        lr_on_plateau.system = sample_system_object()
        lr_on_plateau.best = 5
        lr_on_plateau.wait = 11
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            lr_on_plateau.on_epoch_end(data=self.data)
            log = fake_stdout.getvalue().strip()
            self.assertEqual(log, self.torch_expected_msg)
        with self.subTest('Check learning rate in data'):
            self.assertTrue(math.isclose(self.data[model_name], 0.000100000005, rel_tol=1e-3))

    def test_torch_model_on_epoch_end_lr_wait(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')
        lr_on_plateau = ReduceLROnPlateau(model=model, metric='loss')
        lr_on_plateau.system = sample_system_object()
        lr_on_plateau.best = 12
        lr_on_plateau.on_epoch_end(data=self.data)
        with self.subTest('Check value of wait'):
            self.assertEqual(lr_on_plateau.wait, 0)
        with self.subTest('Check value of best'):
            self.assertEqual(lr_on_plateau.best, 10)
