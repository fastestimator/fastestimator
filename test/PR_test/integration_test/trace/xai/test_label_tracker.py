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

import tensorflow as tf

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.xai import LabelTracker
from fastestimator.util import Data


class TestLabelTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Epoch 1 Data
        batch1_metrics = tf.constant([0.9, 0.8, 0.1, 0.4, 0.9, 0.6, 0.1, 0.6, 0.8, 0.3])
        batch1_labels = tf.constant([2, 1, 0, 1, 0, 1, 1, 1, 0, 0])
        batch2_metrics = tf.constant([0.3, 0.1, 0.0, 0.8, 0.5, 0.5, 0.6, 0.2, 0.5, 1.0])
        batch2_labels = tf.constant([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        batch3_metrics = tf.constant([0.3, 0.4, 0.7, 0.9, 0.3, 0.0, 0.3, 0.7, 0.8, 0.6])
        batch3_labels = tf.constant([2, 2, 2, 2, 2, 0, 2, 2, 2, 2])
        # Epoch 2 Data
        batch4_metrics = tf.constant([0.8, 0.9, 0.6, 0.1, 0.7, 0.3, 0.9, 0.9, 0.4, 0.6])
        batch4_labels = tf.constant([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
        batch5_metrics = tf.constant([0.4, 0.4, 0.9, 0.1, 0.4, 0.9, 0.0, 0.8, 1.0, 0.1])
        batch5_labels = tf.constant([2, 2, 2, 2, 0, 1, 0, 0, 0, 0])
        batch6_metrics = tf.constant([0.3, 0.9, 0.9, 0.5, 0.9, 0.8, 0.6, 0.1, 0.1, 0.2])
        batch6_labels = tf.constant([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

        cls.training_data = [[
            Data({'acc': batch1_metrics, 'y': batch1_labels}),
            Data({'acc': batch2_metrics, 'y': batch2_labels}),
            Data({'acc': batch3_metrics, 'y': batch3_labels})
        ], [Data({'acc': batch4_metrics, 'y': batch4_labels}),
            Data({'acc': batch5_metrics, 'y': batch5_labels}),
            Data({'acc': batch6_metrics, 'y': batch6_labels})
            ]]

        cls.epoch_1_means = {0: 0.49, 1: 0.42, 2: 0.59}
        cls.epoch_2_means = {0: 0.54, 1: 0.66, 2: 0.45}

    def _simulate_training(self, trace: LabelTracker, data: Data):
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
        labeltracker = LabelTracker(label='y', metric='acc', bounds=None, outputs='out')
        system = sample_system_object()
        labeltracker.system = system
        response = Data()
        self._simulate_training(labeltracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that all 3 labels have summaries'):
            self.assertEqual(3, len(response))
        with self.subTest('Check correct mean values (epoch 1)'):
            self.assertDictEqual(self.epoch_1_means,
                                 {elem.name: round(elem.history['train']['acc'][3], 6)
                                  for elem in response})
        with self.subTest('Check correct mean values (epoch 2)'):
            self.assertDictEqual(self.epoch_2_means,
                                 {elem.name: round(elem.history['train']['acc'][6], 6)
                                  for elem in response})

    def test_basic_limiting_labels(self):
        labeltracker = LabelTracker(label='y',
                                    metric='acc',
                                    bounds=None,
                                    label_mapping={'good': 0, 'bad': 1},
                                    outputs='out')
        system = sample_system_object()
        labeltracker.system = system
        response = Data()
        self._simulate_training(labeltracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that all 2 labels have summaries'):
            self.assertEqual(2, len(response))
        with self.subTest('Check correct mean values (epoch 1)'):
            target = {'good': self.epoch_1_means[0], 'bad': self.epoch_1_means[1]}
            self.assertDictEqual(target,
                                 {elem.name: round(elem.history['train']['acc'][3], 6)
                                  for elem in response})
        with self.subTest('Check correct mean values (epoch 2)'):
            target = {'good': self.epoch_2_means[0], 'bad': self.epoch_2_means[1]}
            self.assertDictEqual(target,
                                 {elem.name: round(elem.history['train']['acc'][6], 6)
                                  for elem in response})

    def test_multiple_bounds(self):
        labeltracker = LabelTracker(label='y', metric='acc', bounds=['std', 'range'], outputs='out')
        system = sample_system_object()
        labeltracker.system = system
        response = Data()
        self._simulate_training(labeltracker, response)

        # Check responses
        with self.subTest('Check that a response was written'):
            self.assertIn('out', response)
        with self.subTest('Check that data was written to the system'):
            self.assertIn('out', system.custom_graphs)
        response = response['out']
        with self.subTest('Check consistency of outputs'):
            self.assertEqual(response, system.custom_graphs['out'])
        with self.subTest('Check that all 3 labels have summaries'):
            self.assertEqual(3, len(response))
        with self.subTest('Check that regular mean is not present'):
            for elem in response:
                self.assertNotIn('acc', elem.history['train'])
        with self.subTest('Check that stddev and range are both present'):
            for elem in response:
                self.assertIn('acc ($\\mu \\pm \\sigma$)', elem.history['train'])
                self.assertIn('acc ($min, \\mu, max$)', elem.history['train'])

    def test_save_and_load_state(self):
        def instantiate_system():
            tracker = LabelTracker(label='y', metric='acc', bounds=[None, 'range'], outputs='out')
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
        response = list(loaded_tracker.label_summaries.values())
        with self.subTest('Check that all 3 labels have summaries'):
            self.assertEqual(3, len(response))
        with self.subTest('Check correct mean values (epoch 1)'):
            self.assertDictEqual(self.epoch_1_means,
                                 {elem.name: round(elem.history['train']['acc'][3], 6)
                                  for elem in response})
        with self.subTest('Check correct mean values (epoch 2)'):
            self.assertDictEqual(self.epoch_2_means,
                                 {elem.name: round(elem.history['train']['acc'][6], 6)
                                  for elem in response})
