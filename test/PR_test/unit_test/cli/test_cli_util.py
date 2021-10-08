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

from fastestimator.util.cli_util import parse_cli_to_dictionary


class TestCliUtil(unittest.TestCase):
    def test_parse_cli_to_dictionary(self):
        a = parse_cli_to_dictionary(["--epochs", "5", "--test", "this", "--lr", "0.74"])
        self.assertEqual(a, {'epochs': 5, 'test': 'this', 'lr': 0.74})

    def test_parse_cli_to_dictionary_no_key(self):
        a = parse_cli_to_dictionary(["abc", "def"])
        self.assertEqual(a, {})

    def test_parse_cli_to_dictionary_no_value(self):
        a = parse_cli_to_dictionary(["--abc", "--def"])
        self.assertEqual(a, {"abc": "", "def": ""})

    def test_parse_cli_to_dictionary_consecutive_value(self):
        a = parse_cli_to_dictionary(["--abc", "hello", "world"])
        self.assertEqual(a, {"abc": "helloworld"})
