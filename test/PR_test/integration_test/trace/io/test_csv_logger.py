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
import csv
import os
import tempfile
import unittest
from collections import defaultdict

import pandas as pd
from pandas.testing import assert_frame_equal

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.io import CSVLogger
from fastestimator.util.data import Data


class TestCSVLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data({'mode': 'train', 'epoch': 1})
        cls.df = pd.DataFrame([['train', None, 0]], columns=["mode", "step", "epoch"])
        cls.csv_path = os.path.join(tempfile.mkdtemp(), 'test_csv_logger.csv')

    def test_on_begin(self):
        csvlogger = CSVLogger(filename=self.csv_path)
        csvlogger.on_begin(data=self.data)
        assert_frame_equal(csvlogger.df, pd.DataFrame(columns=["mode", "step", "epoch"]))

    def test_on_epoch_end(self):
        csvlogger = CSVLogger(filename=self.csv_path)
        csvlogger.system = sample_system_object()
        csvlogger.df = pd.DataFrame(columns=["mode", "step", "epoch"])
        csvlogger.on_epoch_end(data=self.data)
        assert_frame_equal(csvlogger.df, self.df, check_dtype=False)

    def test_on_end(self):
        csvlogger = CSVLogger(filename=self.csv_path)
        csvlogger.system = sample_system_object()
        csvlogger.df = pd.DataFrame([['train', None, 0]], columns=["mode", "step", "epoch"])
        # remove csv file previously created if it exists
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        csv_data = []
        # call on_end here and create new csv
        csvlogger.on_end(data=self.data)
        with open(self.csv_path) as f:
            records = csv.DictReader(f)
            for param in records:
                csv_data.append(defaultdict(list, dict(param)))
        with self.subTest('Check if path exists'):
            self.assertTrue(os.path.exists(self.csv_path))
        with self.subTest('Check mode value stored in csv file'):
            self.assertEqual(csv_data[0]['mode'], csvlogger.df['mode'].iloc[0])
        with self.subTest('Check epoch value stored in csv file'):
            self.assertEqual(int(csv_data[0]['epoch']), csvlogger.df['epoch'].iloc[0])
