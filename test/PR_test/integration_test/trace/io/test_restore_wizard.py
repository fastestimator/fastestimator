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

import fastestimator as fe
from fastestimator.backend.load_model import load_model
from fastestimator.backend.save_model import save_model
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch
from fastestimator.trace.io import RestoreWizard
from fastestimator.util.data import Data


def get_model_name(system):
    model_names = []
    for model in system.network.models:
        model_names.append(model.model_name)
    return model_names


class TestRestoreWizard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system_json_path = os.path.join(tempfile.gettempdir(), 'restorewizard')

    def setUp(self):
        self.data = Data({})

    def test_tf_model_on_begin(self):
        restore_wizard = RestoreWizard(directory=self.system_json_path)
        restore_wizard.system = sample_system_object()
        # save state
        for model in restore_wizard.system.network.models:
            save_model(model, save_dir=restore_wizard.directory, save_optimizer=True)
        restore_wizard.system.save_state(json_path=os.path.join(restore_wizard.directory, restore_wizard.system_file))
        restore_wizard.on_begin(data=self.data)
        with self.subTest('Check the restore files directory'):
            self.assertEqual(restore_wizard.directory, self.system_json_path)
        with self.subTest('check data dictionary'):
            self.assertEqual(self.data['epoch'], 0)
        if os.path.exists(self.system_json_path):
            shutil.rmtree(self.system_json_path)

    def test_tf_model_on_epoch_end(self):
        restore_wizard = RestoreWizard(directory=self.system_json_path)
        restore_wizard.system = sample_system_object()
        restore_wizard.on_epoch_end(data=self.data)
        model_names = get_model_name(restore_wizard.system)
        with self.subTest('check json exists'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, 'system.json')))
        with self.subTest('Check if model weights path stored'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, model_names[0] + '.h5')))
        with self.subTest('Check if model optimizer stored'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, model_names[0] + '_opt.pkl')))
        if os.path.exists(self.system_json_path):
            shutil.rmtree(self.system_json_path)

    def test_torch_model_on_begin(self):
        restore_wizard = RestoreWizard(directory=self.system_json_path)
        restore_wizard.system = sample_system_object_torch()
        # save state
        for model in restore_wizard.system.network.models:
            save_model(model, save_dir=restore_wizard.directory, save_optimizer=True)
        restore_wizard.system.save_state(json_path=os.path.join(restore_wizard.directory, restore_wizard.system_file))
        restore_wizard.on_begin(data=self.data)
        with self.subTest('Check the restore files directory'):
            self.assertEqual(restore_wizard.directory, self.system_json_path)
        with self.subTest('check data dictionary'):
            self.assertEqual(self.data['epoch'], 0)
        if os.path.exists(self.system_json_path):
            shutil.rmtree(self.system_json_path)

    def test_torch_model_on_epoch_end(self):
        restore_wizard = RestoreWizard(directory=self.system_json_path)
        restore_wizard.system = sample_system_object_torch()
        restore_wizard.on_epoch_end(data=self.data)
        model_names = get_model_name(restore_wizard.system)
        with self.subTest('check json exists'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, 'system.json')))
        with self.subTest('Check if model weights path stored'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, model_names[0] + '.pt')))
        with self.subTest('Check if model optimizer stored'):
            self.assertTrue(os.path.exists(os.path.join(self.system_json_path, model_names[0] + '_opt.pt')))
        if os.path.exists(self.system_json_path):
            shutil.rmtree(self.system_json_path)
