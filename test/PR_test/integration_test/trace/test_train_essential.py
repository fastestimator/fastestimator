import math
import time
import unittest

import torch

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace import TrainEssential
from fastestimator.util.data import Data


class TestTrainEssential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'loss': 10})
        cls.train_essential = TrainEssential(monitor_names='loss')
        cls.train_essential.system = sample_system_object()
        cls.train_essential.system.log_steps = 5
        cls.train_essential.system.global_step = 10
        cls.train_essential.epoch_start = time.perf_counter() - 500
        cls.train_essential.step_start = time.perf_counter() - 300

    def test_on_begin(self):
        self.train_essential.on_begin(data=self.data)
        with self.subTest('Check number of devices'):
            self.assertEqual(self.data['num_device'], torch.cuda.device_count())
        with self.subTest('Check logging interval'):
            self.assertEqual(self.data['logging_interval'], 5)

    def test_epoch_begin(self):
        self.train_essential.on_epoch_begin(data=self.data)
        with self.subTest('Epoch start time must not be none'):
            self.assertIsNotNone(self.train_essential.epoch_start)
        with self.subTest('Step start time must not be none'):
            self.assertIsNotNone(self.train_essential.step_start)

    def test_on_batch_end(self):
        self.train_essential.on_batch_end(data=self.data)
        with self.subTest('Check steps/sec in data'):
            self.assertIsNotNone(self.data['steps/sec'])
        with self.subTest('Check elapse time list'):
            self.assertEqual(self.train_essential.elapse_times, [])

    def test_on_epoch_end(self):
        self.train_essential.on_epoch_end(data=self.data)
        self.assertIsNotNone(self.data['epoch_time'])

    def test_on_end(self):
        self.train_essential.on_end(data=self.data)
        model_name = self.train_essential.system.network.models[0].model_name
        with self.subTest('Check total time in data'):
            self.assertIsNotNone(self.data['total_time'])
        with self.subTest('Check model learning rate in data dictionary'):
            self.assertTrue(math.isclose(self.data[model_name + '_lr'], 0.001, rel_tol=1e-3))
