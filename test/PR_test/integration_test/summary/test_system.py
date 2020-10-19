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
import os
import shutil
import tempfile
import unittest

import tensorflow as tf

from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch
from fastestimator.trace.trace import Trace


def get_model_names(system):
    model_names = []
    for model in system.network.models:
        model_names.append(model.model_name)
    return model_names


class TestTrace(Trace):
    def __init__(self, var1):
        super().__init__()
        self.var1 = var1


class TestSystem(unittest.TestCase):
    def test_save_and_load_state_torch(self):
        """ `save_state` and `load_state` of an entire system are highly dependent on the implementation of __setstate__
        and __getstate__ of ops, traces, datasets ... etc. The save_state and load_state function should be tested in
        the testing script of that perticular component
        (ex: The test of save_state and load_state with EpochScheduler is located in
             PR_test/integration_test/schedule/test_epoch_scheduler.py)
        """
        system = sample_system_object_torch()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'objects.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.pt')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pt')))

        system = sample_system_object_torch()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(global_step, system.global_step)
            self.assertEqual(epoch_idx, system.epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_save_and_load_state_tf(self):
        """ `save_state` and `load_state` of an entire system are highly dependent on the implementation of __setstate__
        and __getstate__ of ops, traces, datasets ... etc. The save_state and load_state function should be tested in
        the testing script of that perticular component
        (ex: The test of save_state and load_state with EpochScheduler is located in
             PR_test/integration_test/schedule/test_epoch_scheduler.py)
        """
        system = sample_system_object()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'objects.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.h5')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pkl')))

        system = sample_system_object()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(global_step, system.global_step)
            self.assertEqual(epoch_idx, system.epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_shared_variables_within_traces(self):
        save_path = tempfile.mkdtemp()

        system = sample_system_object()
        shared_trace_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_trace_var))
        system.traces.append(TestTrace(shared_trace_var))
        shared_trace_var.assign_add(1)

        system.save_state(save_dir=save_path)

        # Re-initialize
        system = sample_system_object()
        shared_trace_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_trace_var))
        system.traces.append(TestTrace(shared_trace_var))

        system.load_state(load_dir=save_path)

        with self.subTest("Check variable value was re-loaded"):
            self.assertEqual(3, system.traces[-1].var1.numpy())
            self.assertEqual(3, system.traces[-2].var1.numpy())

        with self.subTest("Check that variable is still shared"):
            system.traces[-1].var1.assign(5)
            self.assertEqual(5, system.traces[-1].var1.numpy())
            self.assertEqual(5, system.traces[-2].var1.numpy())

        # If we ever need this edge case to work, we need to make a default __setstate__ method
        # with self.subTest("Check that variable is still linked to outside code"):
        #     shared_trace_var.assign(7)
        #     self.assertEqual(7, system.traces[-1].var1.numpy())
        #     self.assertEqual(7, system.traces[-2].var1.numpy())

    def test_shared_variables_over_object_types(self):
        save_path = tempfile.mkdtemp()

        system = sample_system_object()
        shared_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_var))
        system.network.ops[0].fe_test_var_1 = shared_var
        shared_var.assign_add(1)

        system.save_state(save_dir=save_path)

        # Re-initialize
        system = sample_system_object()
        shared_var = tf.Variable(initial_value=2, trainable=True)
        system.traces.append(TestTrace(shared_var))
        system.network.ops[0].fe_test_var_1 = shared_var

        system.load_state(load_dir=save_path)

        with self.subTest("Check variable value was re-loaded"):
            self.assertEqual(3, system.traces[-1].var1.numpy())
            self.assertEqual(3, system.network.ops[0].fe_test_var_1.numpy())

        with self.subTest("Check that variable is still shared"):
            system.traces[-1].var1.assign(5)
            self.assertEqual(5, system.traces[-1].var1.numpy())
            self.assertEqual(5, system.network.ops[0].fe_test_var_1.numpy())

        # If we ever need this edge case to work, we need to make a default __setstate__ method
        # with self.subTest("Check that variable is still linked to outside code"):
        #     shared_trace_var.assign(7)
        #     self.assertEqual(7, system.traces[-1].var1.numpy())
        #     self.assertEqual(7, system.network.ops[0].fe_test_var_1.numpy())
