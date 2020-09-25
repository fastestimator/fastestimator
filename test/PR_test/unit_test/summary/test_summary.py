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


class TestSummary(unittest.TestCase):
    def test_merge(self):
        summary1 = fe.summary.Summary(name='test1')
        summary1.history['e1']['metrics'] = {'acc': 0.9}

        summary2 = fe.summary.Summary(name='test2')
        summary2.history['e1']['metrics'] = {'acc': 0.8}

        summary1.merge(summary2)

        self.assertEqual(summary1.history['e1']['metrics']['acc'], 0.8)
