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

import pandas as pd

import fastestimator as fe


class TestCSVDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        data = {'x': ['a1.txt', 'a2.txt', 'b1.txt', 'b2.txt'], 'y': [0, 0, 1, 1]}
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(tmpdirname, 'data.csv'), index=False)

        dataset = fe.dataset.CSVDataset(file_path=os.path.join(tmpdirname, 'data.csv'))

        self.assertEqual(len(dataset), 4)
