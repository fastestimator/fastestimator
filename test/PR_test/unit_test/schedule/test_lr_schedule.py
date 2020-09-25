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
import math
import unittest

from fastestimator.schedule.lr_shedule import cosine_decay


class TestLRSchedule(unittest.TestCase):
    def test_cosine_decay(self):
        learning_rate = cosine_decay(time=5, cycle_length=10, init_lr=0.01, min_lr=0.0)
        self.assertEqual(learning_rate, 0.005)

    def test_cosine_decay_cycle(self):
        learning_rate = cosine_decay(time=1001, cycle_length=1000, init_lr=0.01, min_lr=0.0)
        self.assertTrue(math.isclose(learning_rate, 0.01, rel_tol=1e-3))

    def test_cosine_decay_multiplier_mid_cycle(self):
        learning_rate = cosine_decay(5, cycle_length=10, init_lr=0.01, min_lr=0.0, cycle_multiplier=2)
        self.assertEqual(learning_rate, 0.005)

    def test_cosine_decay_multiplier_end_cycle(self):
        learning_rate = cosine_decay(10, cycle_length=10, init_lr=0.01, min_lr=0.0, cycle_multiplier=2)
        self.assertEqual(learning_rate, 0.0)
