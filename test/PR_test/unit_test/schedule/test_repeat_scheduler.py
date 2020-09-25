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
import unittest

from fastestimator.schedule import RepeatScheduler


class TestRepeatScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_data = [1, 2, 2]
        cls.scheduler = RepeatScheduler(cls.input_data)

    def test_repeat_scheduler_current_value(self):
        self.assertEqual(self.scheduler.get_current_value(epoch=2), 2)

    def test_repeat_scheduler_all_values(self):
        self.assertEqual(self.scheduler.get_all_values(), self.input_data)
