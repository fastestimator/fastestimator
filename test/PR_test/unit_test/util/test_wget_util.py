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


class TestWgetUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = " 10% [......                                                        ] 0.00 / 0.00 MB"

    def test_bar_custom(self):
        self.assertEqual(fe.util.wget_util.bar_custom(10, 100), self.op)
