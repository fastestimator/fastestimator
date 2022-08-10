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

    @classmethod
    def setUpClass(cls) -> None:
        tmpdirname = tempfile.mkdtemp()
        data = {'idx': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                'mode': ['train', 'eval', 'test', 'train', 'eval', 'test', 'train', 'eval', 'test'],
                'type': [0, 1, 2, 2, 0, 1, 1, 0, None]
                }
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(tmpdirname, 'data.csv'), index=False)
        cls.csv_path = os.path.join(tmpdirname, 'data.csv')

    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        data = {'x': ['a1.txt', 'a2.txt', 'b1.txt', 'b2.txt'], 'y': [0, 0, 1, 1]}
        df = pd.DataFrame(data=data)
        df.to_csv(os.path.join(tmpdirname, 'data.csv'), index=False)

        dataset = fe.dataset.CSVDataset(file_path=os.path.join(tmpdirname, 'data.csv'))

        self.assertEqual(len(dataset), 4)

    def test_filter_single_key_unwrapped(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path, include_if={'mode': 'train'})
        self.assertSetEqual(set(dataset['idx']), {0, 3, 6})
        self.assertSetEqual(set(dataset['mode']), {'train'})

    def test_filter_single_key_wrapped(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path, include_if={'mode': ['train']})
        self.assertSetEqual(set(dataset['idx']), {0, 3, 6})
        self.assertSetEqual(set(dataset['mode']), {'train'})

    def test_filter_single_key_multi_val(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path, include_if={'mode': ('train', 'eval')})
        self.assertSetEqual(set(dataset['idx']), {0, 1, 3, 4, 6, 7})
        self.assertSetEqual(set(dataset['mode']), {'train', 'eval'})

    def test_filter_multi_key(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path, include_if={'mode': 'train', 'type': 0})
        self.assertSetEqual(set(dataset['idx']), {0})
        self.assertSetEqual(set(dataset['mode']), {'train'})

    def test_filter_multi_key_multival(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path, include_if={'mode': 'train', 'type': {0, 1}})
        self.assertSetEqual(set(dataset['idx']), {0, 6})
        self.assertSetEqual(set(dataset['mode']), {'train'})

    def test_filter_with_raw_None(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if={'mode': 'test', 'type': None},
                                        fill_na=None)
        self.assertSetEqual(set(dataset['idx']), {8})

    def test_filter_with_wrapped_None(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if={'mode': 'test', 'type': [None]},
                                        fill_na=None)
        self.assertSetEqual(set(dataset['idx']), {8})

    def test_filter_with_multi_None(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if={'mode': 'test', 'type': [2, None]},
                                        fill_na=None)
        self.assertSetEqual(set(dataset['idx']), {2, 8})

    def test_filter_query_string(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if='type >= 1')
        self.assertSetEqual(set(dataset['idx']), {1, 2, 3, 5, 6})

    def test_filter_function_one_inp(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if=lambda mode: mode == 'test')
        self.assertSetEqual(set(dataset['idx']), {2, 5, 8})

    def test_filter_function_two_inp(self):
        dataset = fe.dataset.CSVDataset(file_path=self.csv_path,
                                        include_if=lambda mode, idx: mode == 'test' or idx % 2 == 1)
        self.assertSetEqual(set(dataset['idx']), {1, 2, 3, 5, 7, 8})

    def test_filter_function_bad_col(self):
        with self.assertRaises(ValueError):
            fe.dataset.CSVDataset(file_path=self.csv_path,
                                  include_if=lambda mode, idf: mode == 'test' or idf % 2 == 1)
