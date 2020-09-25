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
from unittest.mock import Mock

import fastestimator as fe


class TestSystem(unittest.TestCase):
    def test_system_save_state_load_state(self):
        system = fe.summary.System(pipeline=Mock(), network=Mock(), traces=Mock())
        global_step = 100
        epoch_idx = 10
        file_path = os.path.join(tempfile.mkdtemp(), "test.json")
        system.global_step = global_step
        system.epoch_idx = epoch_idx

        with self.subTest("check save_state dumped file"):
            system.save_state(file_path)
            self.assertTrue(os.path.exists(file_path))

        system.global_step = 0
        system.epoch_idx = 0
        with self.subTest("check state after load_state"):
            system.load_state(file_path)
            self.assertEqual(system.global_step, global_step)
            self.assertEqual(system.epoch_idx, epoch_idx)
