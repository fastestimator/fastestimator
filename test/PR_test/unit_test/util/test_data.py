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

import fastestimator as fe


class TestData(unittest.TestCase):
    def setUp(self):
        self.d = fe.util.Data({"a": 0, "b": 1, "c": 2})

    def test_write_with_log(self):
        self.d.write_with_log("d", 3)
        self.assertEqual(self.d.read_logs(), {'d': 3})

    def test_write_without_log(self):
        self.d.write_without_log("e", 5)
        self.assertEqual(self.d.read_logs(), {})

    def test_read_logs(self):
        self.d.write_with_log("d", 3)
        self.d.write_with_log("a", 4)
        self.assertEqual(self.d.read_logs(), {"d": 3, "a": 4})
