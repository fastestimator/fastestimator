#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import tempfile
import unittest
from collections import defaultdict
import tensorflow as tf

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.xai import InstanceTracker
from fastestimator.util import Data


class TestInstanceTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Epoch 1 Data
        batch1_metrics = tf.constant([0.9, 0.8, 0.1, 0.4, 0.9, 0.6, 0.1, 0.6, 0.8, 0.3])
        batch1_idx = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        batch2_metrics = tf.constant([0.3, 0.1, 0.0, 0.8, 0.5, 0.5, 0.6, 0.2, 0.5, 1.0])
        batch2_idx = tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        batch3_metrics = tf.constant([0.3, 0.4, 0.7, 0.9, 0.3, 0.0, 0.3, 0.7, 0.8, 0.6])
        batch3_idx = tf.constant([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
        # Epoch 2 Data
        batch4_metrics = tf.constant([0.8, 0.9, 0.6, 0.1, 0.7, 0.3, 0.9, 0.9, 0.4, 0.6])
        batch4_idx = tf.constant([21, 2, 18, 3, 15, 22, 12, 27, 23, 9])
        batch5_metrics = tf.constant([0.4, 0.4, 0.9, 0.1, 0.4, 0.9, 0.0, 0.8, 1.0, 0.1])
        batch5_idx = tf.constant([20, 5, 28, 8, 6, 25, 11, 13, 7, 16])
        batch6_metrics = tf.constant([0.3, 0.9, 0.9, 0.5, 0.9, 0.8, 0.6, 0.1, 0.1, 0.2])
        batch6_idx = tf.constant([1, 19, 24, 10, 0, 29, 17, 14, 4, 26])

        cls.training_data = [[
            Data({'ce': batch1_metrics, 'idx': batch1_idx}),
            Data({'ce': batch2_metrics, 'idx': batch2_idx}),
            Data({'ce': batch3_metrics, 'idx': batch3_idx})
        ], [Data({'ce': batch4_metrics, 'idx': batch4_idx}),
            Data({'ce': batch5_metrics, 'idx': batch5_idx}),
            Data({'ce': batch6_metrics, 'idx': batch6_idx})
            ]]

    def _simulate_training(self, trace: InstanceTracker, data: Data):
        trace.on_begin(Data())
        system = trace.system
        global_step = 0
        for epoch_idx, epoch in enumerate(self.training_data, start=1):
            system.epoch_idx = epoch_idx
            trace.on_epoch_begin(Data())
            for batch_idx, batch in enumerate(epoch, start=1):
                system.batch_idx = batch_idx
                global_step += 1
                system.global_step = global_step
                trace.on_batch_begin(Data())
                trace.on_batch_end(batch)
            trace.on_epoch_end(Data())
        trace.on_end(data)

    def test_basic_happy_path(self):
        instance_tracker = InstanceTracker(index='idx', metric='ce', outputs='out', mode='train')
        system = sample_system_object()
        instance_tracker.system = system
        response = Data()
        self._simulate_training(instance_tracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that 10 indices were tracked'):
            self.assertEqual(10, len(response))
        recorded_indices = {summary.name for summary in response}
        with self.subTest('Check that 5 min keys were tracked'):
            min_keys = [3, 4, 8, 11, 14]
            for key in min_keys:
                self.assertIn(key, recorded_indices)
        with self.subTest('Check that 5 max keys were tracked'):
            max_keys = [7, 8, 24, 25, 27, 28]
            for key in max_keys:
                self.assertIn(key, recorded_indices)

    def test_specific_indices(self):
        instance_tracker = InstanceTracker(index='idx',
                                           metric='ce',
                                           n_max_to_keep=0,
                                           n_min_to_keep=0,
                                           list_to_keep=[1, 5, 17],
                                           outputs='out',
                                           mode='train')
        system = sample_system_object()
        instance_tracker.system = system
        response = Data()
        self._simulate_training(instance_tracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that 3 indices were tracked'):
            self.assertEqual(3, len(response))
        recorded_indices = {summary.name for summary in response}
        with self.subTest('Check that the correct keys were tracked'):
            target_keys = [1, 5, 17]
            for key in target_keys:
                self.assertIn(key, recorded_indices)

    def test_min_max_and_specific(self):
        instance_tracker = InstanceTracker(index='idx',
                                           metric='ce',
                                           list_to_keep=[1, 5, 17],
                                           outputs='out',
                                           mode='train')
        system = sample_system_object()
        instance_tracker.system = system
        response = Data()
        self._simulate_training(instance_tracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that 13 indices were tracked'):
            self.assertEqual(13, len(response))
        recorded_indices = {summary.name for summary in response}
        with self.subTest('Check that 5 min keys were tracked'):
            min_keys = [3, 4, 8, 11, 14]
            for key in min_keys:
                self.assertIn(key, recorded_indices)
        with self.subTest('Check that 5 max keys were tracked'):
            max_keys = [7, 8, 24, 25, 27, 28]
            for key in max_keys:
                self.assertIn(key, recorded_indices)
        with self.subTest('Check that the specified indices were tracked'):
            target_keys = [1, 5, 17]
            for key in target_keys:
                self.assertIn(key, recorded_indices)

    def test_save_and_load_state(self):
        def instantiate_system():
            tracker = InstanceTracker(index='idx',
                                      metric='ce',
                                      n_max_to_keep=0,
                                      n_min_to_keep=0,
                                      list_to_keep=[1, 5, 17],
                                      outputs='out',
                                      mode='train')
            system = sample_system_object()
            system.traces.append(tracker)
            tracker.system = system
            return system, tracker

        system, tracker = instantiate_system()

        # Make some changes
        response = Data()
        self._simulate_training(tracker, response)

        # Save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # reinstantiate system and load the state
        system, tracker = instantiate_system()
        system.load_state(save_path)

        loaded_tracker = system.traces[-1]
        response = loaded_tracker.index_history
        with self.subTest('Check that the restored container has the correct type'):
            self.assertTrue(isinstance(response, defaultdict))
        with self.subTest('Check that the mode was saved'):
            self.assertIn('train', response)
        response = response['train']
        with self.subTest('Check that all 3 labels have summaries'):
            self.assertEqual(3, len(response))
        with self.subTest('Check that the specified indices were restored'):
            target_keys = [1, 5, 17]
            for key in target_keys:
                self.assertIn(key, response)
        response = response[1]
        with self.subTest('Check that values were restored'):
            self.assertEqual(0.8, round(response[0][1], 5))
            self.assertEqual(0.3, round(response[1][1], 5))
