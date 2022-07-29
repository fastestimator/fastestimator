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
import tempfile
import unittest

import tensorflow as tf

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.io import BatchDisplay
from fastestimator.trace.trace import Freq
from fastestimator.util import Data


class TestBatchDisplay(unittest.TestCase):
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

    def _simulate_training(self, trace: BatchDisplay, data: Data):
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

    def test_save_and_load_state(self):
        def instantiate_system():
            trace = BatchDisplay(image='x',
                                 frequency='5e',
                                 save_dir='.',
                                 mode='train')
            system = sample_system_object()
            system.traces.append(trace)
            trace.system = system
            return system, trace

        system, trace = instantiate_system()

        # Make some changes
        response = Data()
        self._simulate_training(trace, response)

        # Save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_dir=save_path)

        # re-instantiate system and load the state
        system, _ = instantiate_system()
        system.load_state(save_path)

        loaded_trace = system.traces[-1]
        with self.subTest('Check that the restored frequency has the correct type'):
            self.assertTrue(isinstance(loaded_trace.frequency, Freq))
        with self.subTest('Check that the frequency was saved'):
            self.assertEqual(loaded_trace.frequency.is_step, False)
            self.assertEqual(loaded_trace.frequency.freq, 5)
