import math
import unittest

import fastestimator as fe
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
