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

import fastestimator as fe
import numpy as np
from fastestimator.architecture.pytorch import LeNet as LeNet_torch
from fastestimator.architecture.tensorflow import LeNet as LeNet_tf
from fastestimator.backend import get_lr
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.test.unittest_util import MultiLayerTorchModel, one_layer_tf_model, sample_system_object
from fastestimator.trace.adapt import LRScheduler
from fastestimator.util.data import Data


class TestLRScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({})

    def setUp(self):
        self.tf_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        self.torch_model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn='adam')

    def create_estimator_for_arc(self, model, use_eval, axis):
        train_data, eval_data = mnist.load_data()
        pipeline = fe.Pipeline(train_data=train_data,
                               eval_data=eval_data if use_eval else None,
                               batch_size=8,
                               ops=[ExpandDims(inputs="x", outputs="x", axis=axis), Minmax(inputs="x", outputs="x")])
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])
        estimator = fe.Estimator(pipeline=pipeline,
                                 network=network,
                                 epochs=2,
                                 traces=LRScheduler(model=model, lr_fn="arc"),
                                 max_train_steps_per_epoch=10)
        return estimator

    def test_arc_frequency_small_epochs(self):
        sample_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        sample_model.loss_name = {"loss"}
        lr_scheduler = LRScheduler(model=sample_model, lr_fn="arc")
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.total_epochs = 6
        lr_scheduler.on_begin(data=self.data)
        self.assertEqual(lr_scheduler.frequency, 1)

    def test_arc_frequency_mid_epochs(self):
        sample_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        sample_model.loss_name = {"loss"}
        lr_scheduler = LRScheduler(model=sample_model, lr_fn="arc")
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.total_epochs = 63
        lr_scheduler.on_begin(data=self.data)
        self.assertEqual(lr_scheduler.frequency, 6)

    def test_arc_frequency_large_epochs(self):
        sample_model = fe.build(model_fn=one_layer_tf_model, optimizer_fn='adam')
        sample_model.loss_name = {"loss"}
        lr_scheduler = LRScheduler(model=sample_model, lr_fn="arc")
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.total_epochs = 1000
        lr_scheduler.on_begin(data=self.data)
        self.assertEqual(lr_scheduler.frequency, 10)

    def test_tf_model_on_epoch_begin(self):
        lr_scheduler = LRScheduler(model=self.tf_model,
                                   lr_fn=lambda epoch: fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.epoch_idx = 3
        lr_scheduler.on_epoch_begin(data=self.data)
        self.assertTrue(math.isclose(self.tf_model.optimizer.lr.numpy(), 0.0009999973, rel_tol=1e-5))

    def test_tf_model_on_batch_begin(self):
        lr_scheduler = LRScheduler(model=self.tf_model,
                                   lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.global_step = 3
        lr_scheduler.on_batch_begin(data=self.data)
        self.assertTrue(math.isclose(self.tf_model.optimizer.lr.numpy(), 0.0009999973, rel_tol=1e-5))

    def test_tf_model_on_batch_end(self):
        model_name = self.tf_model.model_name + '_lr'
        lr_scheduler = LRScheduler(model=self.tf_model,
                                   lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.global_step = 3
        lr_scheduler.system.log_steps = 1
        lr_scheduler.on_batch_end(data=self.data)
        self.assertTrue(math.isclose(self.data[model_name], 0.001, rel_tol=1e-3))

    def test_tf_model_arc_train_eval(self):
        model_tf = fe.build(model_fn=LeNet_tf, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=True, axis=-1)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_tf_model_arc_train_only(self):
        model_tf = fe.build(model_fn=LeNet_tf, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=False, axis=-1)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = np.round(lr_after / lr_before, 3)
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_torch_model_on_epoch_begin(self):
        lr_scheduler = LRScheduler(model=self.torch_model,
                                   lr_fn=lambda epoch: fe.schedule.cosine_decay(epoch, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.epoch_idx = 3
        lr_scheduler.on_epoch_begin(data=self.data)
        new_lr = list(self.torch_model.optimizer.param_groups)[0]['lr']
        self.assertTrue(math.isclose(new_lr, 0.0009999993, rel_tol=1e-5))

    def test_torch_model_on_batch_begin(self):
        lr_scheduler = LRScheduler(model=self.torch_model,
                                   lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.global_step = 3
        lr_scheduler.on_batch_begin(data=self.data)
        new_lr = list(self.torch_model.optimizer.param_groups)[0]['lr']
        self.assertTrue(math.isclose(new_lr, 0.0009999993, rel_tol=1e-5))

    def test_torch_model_on_batch_end(self):
        model_name = self.torch_model.model_name + '_lr'
        lr_scheduler = LRScheduler(model=self.torch_model,
                                   lr_fn=lambda step: fe.schedule.cosine_decay(step, cycle_length=3750, init_lr=1e-3))
        lr_scheduler.system = sample_system_object()
        lr_scheduler.system.global_step = 3
        lr_scheduler.system.log_steps = 1
        lr_scheduler.on_batch_end(data=self.data)
        self.assertTrue(math.isclose(self.data[model_name], 0.001, rel_tol=1e-3))

    def test_torch_model_arc_train_eval(self):
        model_tf = fe.build(model_fn=LeNet_torch, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=True, axis=0)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)

    def test_torch_model_arc_train_only(self):
        model_tf = fe.build(model_fn=LeNet_torch, optimizer_fn="adam")
        lr_before = get_lr(model=model_tf)
        estimator = self.create_estimator_for_arc(model_tf, use_eval=False, axis=0)
        estimator.fit()
        lr_after = get_lr(model=model_tf)
        lr_ratio = lr_after / lr_before
        increased = math.isclose(lr_ratio, 1.618, rel_tol=1e-5)
        constant = math.isclose(lr_ratio, 1.0, rel_tol=1e-5)
        decreased = math.isclose(lr_ratio, 0.618, rel_tol=1e-5)
        self.assertTrue(increased or constant or decreased)
