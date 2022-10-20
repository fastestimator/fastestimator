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

import numpy as np

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.io import CSVLogger
from fastestimator.util.data import Data


class TestCSVLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_root = tempfile.mkdtemp()

    @staticmethod
    def _run_fake_training(**csv_args):
        csvlogger = CSVLogger(**csv_args)
        system = sample_system_object()
        system.epoch_idx = 1
        system.global_step = 1
        csvlogger.system = system

        batch_data = Data(batch_data={"idx": ['a', 'b', 'c'],
                                      "x": np.ones((3, 5, 5)),
                                      "y": np.ones((3, 1)),
                                      "ce": 12.5})
        batch_data.write_per_instance_log(key="dice", value=[0.1, 0.2, 0.3])

        epoch_data = Data()
        epoch_data.write_with_log("ce", 12.5)

        csvlogger.on_begin(Data())
        csvlogger.on_epoch_begin(Data())
        csvlogger.on_batch_begin(Data())
        csvlogger.on_batch_end(batch_data)
        csvlogger.on_epoch_end(epoch_data)
        csvlogger.on_end(Data())

    def test_basic_flow(self):
        csv_path = os.path.join(self.csv_root, 'basic.csv')
        self._run_fake_training(filename=csv_path)
        with self.subTest('Check that file was generated'):
            self.assertTrue(os.path.exists(csv_path))

        rows = []
        with open(csv_path) as f:
            records = csv.DictReader(f)
            for row in records:
                rows.append(row)

        with self.subTest('Check number of rows'):
            self.assertEqual(len(rows), 1)

        with self.subTest('Check columns'):
            self.assertEqual(rows[0].keys(), {'mode', 'step', 'epoch', 'ce'})

        with self.subTest('Check values'):
            row = rows[0]
            self.assertEqual(row['mode'], 'train')
            self.assertEqual(row['step'], '1')
            self.assertEqual(row['epoch'], '1')
            self.assertEqual(row['ce'], '12.5')

    def test_per_instance(self):
        csv_path = os.path.join(self.csv_root, 'per_instance.csv')
        self._run_fake_training(filename=csv_path, instance_id_key='idx')
        with self.subTest('Check that file was generated'):
            self.assertTrue(os.path.exists(csv_path))

        rows = []
        with open(csv_path) as f:
            records = csv.DictReader(f)
            for row in records:
                rows.append(row)

        with self.subTest('Check number of rows'):
            self.assertEqual(len(rows), 4)

        with self.subTest('Check columns'):
            self.assertEqual(rows[0].keys(), {'instance_id', 'mode', 'step', 'epoch', 'ce', 'dice'})

        instance_ids = {'': {'instance_id': '',
                             'mode': 'train',
                             'step': '1',
                             'epoch': '1',
                             'ce': '12.5',
                             'dice': ''},
                        'a': {'instance_id': 'a',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'dice': '0.1'},
                        'b': {'instance_id': 'b',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'dice': '0.2'},
                        'c': {'instance_id': 'c',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'dice': '0.3'}}
        with self.subTest('Check values'):
            for row in rows:
                self.assertTrue('instance_id' in row)
                target = instance_ids.pop(row['instance_id'])
                self.assertDictEqual(row, target)
        self.assertEqual(len(instance_ids), 0)

    def test_per_instance_extra_key(self):
        csv_path = os.path.join(self.csv_root, 'per_instance.csv')
        self._run_fake_training(filename=csv_path, instance_id_key='idx', monitor_names=["*", "y"])
        with self.subTest('Check that file was generated'):
            self.assertTrue(os.path.exists(csv_path))

        rows = []
        with open(csv_path) as f:
            records = csv.DictReader(f)
            for row in records:
                rows.append(row)

        with self.subTest('Check number of rows'):
            self.assertEqual(len(rows), 4)

        with self.subTest('Check columns'):
            self.assertEqual(rows[0].keys(), {'instance_id', 'mode', 'step', 'epoch', 'ce', 'y', 'dice'})

        instance_ids = {'': {'instance_id': '',
                             'mode': 'train',
                             'step': '1',
                             'epoch': '1',
                             'ce': '12.5',
                             'y': '',
                             'dice': ''},
                        'a': {'instance_id': 'a',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'y': '[1.]',
                              'dice': '0.1'},
                        'b': {'instance_id': 'b',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'y': '[1.]',
                              'dice': '0.2'},
                        'c': {'instance_id': 'c',
                              'mode': 'train',
                              'step': '1',
                              'epoch': '1',
                              'ce': '',
                              'y': '[1.]',
                              'dice': '0.3'}}
        with self.subTest('Check values'):
            for row in rows:
                self.assertTrue('instance_id' in row)
                target = instance_ids.pop(row['instance_id'])
                self.assertDictEqual(row, target)
        self.assertEqual(len(instance_ids), 0)
