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
import unittest

import fastestimator as fe


class TestPickleDataset(unittest.TestCase):
    def test_dataset(self):
        test_data = fe.dataset.PickleDataset(os.path.abspath(os.path.join(__file__, "..", "resources", "dummy.pkl")))

        self.assertEqual(len(test_data), 5)
