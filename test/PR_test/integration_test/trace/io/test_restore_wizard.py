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
import tempfile
import unittest

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.io import RestoreWizard
from fastestimator.util.data import Data


def get_model_name(system):
    model_names = []
    for model in system.network.models:
        model_names.append(model.model_name)
    return model_names


class TestRestoreWizard(unittest.TestCase):
    def test_save(self):
        save_path = tempfile.mkdtemp()
        restore_wizard = RestoreWizard(directory=save_path)
        restore_wizard.system = sample_system_object()
        restore_wizard.on_begin(Data())
        restore_wizard.on_epoch_end(Data())
        with self.subTest("Check Saved Files (1)"):
            self.assertTrue(os.path.exists(os.path.join(save_path, 'key.txt')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'A')))
        with self.subTest("Check Key is Correct (1)"):
            with open(os.path.join(save_path, 'key.txt'), 'r') as file:
                key = file.readline()
                self.assertEqual(key, "A")
        restore_wizard.on_epoch_end(Data())
        with self.subTest("Check Saved Files (2)"):
            self.assertTrue(os.path.exists(os.path.join(save_path, 'key.txt')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'B')))
        with self.subTest("Check Key is Correct (2)"):
            with open(os.path.join(save_path, 'key.txt'), 'r') as file:
                key = file.readline()
                self.assertEqual(key, "B")
        restore_wizard.on_epoch_end(Data())
        with self.subTest("Check Saved Files (3)"):
            self.assertTrue(os.path.exists(os.path.join(save_path, 'key.txt')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'A')))
        with self.subTest("Check Key is Correct (3)"):
            with open(os.path.join(save_path, 'key.txt'), 'r') as file:
                key = file.readline()
                self.assertEqual(key, "A")
        restore_wizard.on_epoch_end(Data())
        with self.subTest("Check Saved Files (4)"):
            self.assertTrue(os.path.exists(os.path.join(save_path, 'key.txt')))
            self.assertTrue(os.path.exists(os.path.join(save_path, 'B')))
        with self.subTest("Check Key is Correct (4)"):
            with open(os.path.join(save_path, 'key.txt'), 'r') as file:
                key = file.readline()
                self.assertEqual(key, "B")

    def test_restore(self):
        save_path = tempfile.mkdtemp()
        global_step = 100
        epoch_idx = 10

        restore_wizard = RestoreWizard(directory=save_path)
        restore_wizard.system = sample_system_object()
        restore_wizard.on_begin(Data())
        restore_wizard.system.global_step = global_step
        restore_wizard.system.epoch_idx = epoch_idx
        restore_wizard.on_epoch_end(Data())

        restore_wizard = RestoreWizard(directory=save_path)
        restore_wizard.system = sample_system_object()
        data = Data()
        restore_wizard.on_begin(data)
        with self.subTest("Check print message"):
            self.assertEqual(data['epoch'], 10)
        with self.subTest("Check system variables"):
            self.assertEqual(restore_wizard.system.global_step, global_step)
            self.assertEqual(restore_wizard.system.epoch_idx, epoch_idx)
