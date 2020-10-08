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

from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch


def get_model_names(system):
    model_names = []
    for model in system.network.models:
        model_names.append(model.model_name)
    return model_names


class TestSystem(unittest.TestCase):
    def test_save_and_load_state_torch(self):
        system = sample_system_object_torch()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'ds.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'nops.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'summary.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'tops.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'traces.pkl')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.pt')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pt')))

        system = sample_system_object_torch()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(system.global_step, global_step)
            self.assertEqual(system.epoch_idx, epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_save_and_load_state_tf(self):
        system = sample_system_object()
        model_names = get_model_names(system)
        global_step = 100
        epoch_idx = 10
        save_path = tempfile.mkdtemp()

        system.global_step = global_step
        system.epoch_idx = epoch_idx
        with self.subTest("Check state files were created"):
            system.save_state(save_dir=save_path)
            self.assertTrue(os.path.exists(os.path.join(save_path, 'ds.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'nops.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'summary.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'system.json')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'tops.pkl')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'traces.pkl')))
            for model_name in model_names:
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}.h5')))
                self.assertTrue(os.path.exists(os.path.join(save_path, f'{model_name}_opt.pkl')))

        system = sample_system_object()
        with self.subTest("Check that state loads properly"):
            system.load_state(save_path)
            self.assertEqual(system.global_step, global_step)
            self.assertEqual(system.epoch_idx, epoch_idx)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
