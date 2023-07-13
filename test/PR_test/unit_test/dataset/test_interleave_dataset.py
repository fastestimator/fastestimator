# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.dataset.batch_dataset import BatchDataset
from fastestimator.dataset.extend_dataset import ExtendDataset
from fastestimator.dataset.interleave_dataset import InterleaveDataset
from fastestimator.dataset.numpy_dataset import NumpyDataset


class TestInterleaveDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data1 = {"x": [x for x in range(1, 11)], "ds_id": [0 for _ in range(10)]}
        self.data2 = {"x": [x * 10 for x in range(1, 21)], "ds_id": [1 for _ in range(20)]}
        self.data3 = {"x": [x for x in range(1, 11)], "ds_id": [2 for _ in range(10)]}
        self.data4 = {"x": [x for x in range(1, 11)], "ds_id": [3 for _ in range(10)]}

    def test_list_dataset_default_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets=[ds1, ds2])
        self.assertEqual(dataset[0][0]["ds_id"], 0)
        self.assertEqual(dataset[1][0]["ds_id"], 1)
        self.assertEqual(dataset[4][0]["ds_id"], 0)
        self.assertEqual(dataset[9][0]["ds_id"], 1)

    def test_list_dataset_specific_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets=[ds1, ds2], pattern=[0, 1, 1])
        self.assertEqual(dataset[0][0]["ds_id"], 0)
        self.assertEqual(dataset[1][0]["ds_id"], 1)
        self.assertEqual(dataset[2][0]["ds_id"], 1)
        self.assertEqual(dataset[3][0]["ds_id"], 0)

    def test_list_dataset_incorrect_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        with self.assertRaises(AssertionError):
            InterleaveDataset(datasets=[ds1, ds2], pattern=[0, 1, 2])

    def test_dict_dataset_default_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        self.assertEqual(dataset[0][0]["ds_id"], 0)
        self.assertEqual(dataset[1][0]["ds_id"], 1)
        self.assertEqual(dataset[4][0]["ds_id"], 0)
        self.assertEqual(dataset[9][0]["ds_id"], 1)

    def test_dict_dataset_specific_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2}, pattern=['ds2', 'ds2', 'ds1'])
        self.assertEqual(dataset[0][0]["ds_id"], 1)
        self.assertEqual(dataset[1][0]["ds_id"], 1)
        self.assertEqual(dataset[2][0]["ds_id"], 0)
        self.assertEqual(dataset[3][0]["ds_id"], 1)

    def test_dict_dataset_incorrect_pattern_int(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        with self.assertRaises(AssertionError):
            InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2}, pattern=[0, 1])

    def test_dict_dataset_incorrect_pattern_str(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        with self.assertRaises(AssertionError):
            InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2}, pattern=['ds2', 'ds3'])

    def test_interleave_batch_dataset_hybrid(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        batch_ds = BatchDataset(datasets=[ds1, ds2], num_samples=[1, 1])
        ds_3 = NumpyDataset(self.data3)
        dataset = InterleaveDataset(datasets=[batch_ds, ds_3])
        dataset.set_batch_sizes([1, 2])
        self.assertEqual(len(dataset[0]), 2)
        self.assertEqual(len(dataset[1]), 2)

    def test_interleave_batch_dataset_only(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        batch_ds1 = BatchDataset(datasets=[ds1, ds2], num_samples=[1, 1])
        ds3, ds4 = NumpyDataset(self.data3), NumpyDataset(self.data4)
        batch_ds2 = BatchDataset(datasets=[ds3, ds4], num_samples=[1, 1])
        dataset = InterleaveDataset(datasets=[batch_ds1, batch_ds2])
        dataset.set_batch_sizes([2, 3])
        self.assertEqual(len(dataset[0]), 4)
        self.assertEqual(len(dataset[1]), 6)

    def test_length(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        self.assertEqual(len(dataset), 20)

    def test_length_extend(self):
        ds1, ds2 = ExtendDataset(NumpyDataset(self.data1), spoof_length=100), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        self.assertEqual(len(dataset), 40)
        dataset.set_batch_sizes([5, 1])
        self.assertEqual(len(dataset), 40)
        self.assertEqual(len(dataset[38]), 5)
        self.assertEqual(len(dataset[39]), 1)

    def test_length_contract(self):
        ds1, ds2 = NumpyDataset(self.data1), ExtendDataset(NumpyDataset(self.data2), spoof_length=5)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        self.assertEqual(len(dataset), 10)
        dataset.set_batch_sizes([2, 2])
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset[2]), 2)
        self.assertEqual(len(dataset[3]), 2)

    def test_split_with_batch_ds(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        batch_ds1 = BatchDataset(datasets=[ds1, ds2], num_samples=[1, 1])
        ds3, ds4 = NumpyDataset(self.data3), NumpyDataset(self.data4)
        batch_ds2 = BatchDataset(datasets=[ds3, ds4], num_samples=[1, 1])
        dataset = InterleaveDataset(datasets=[batch_ds1, batch_ds2])
        self.assertEqual(len(dataset), 20)
        dataset2 = dataset.split(0.5)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset2), 10)

    def test_split_without_batch_ds(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        self.assertEqual(len(dataset), 20)
        dataset2 = dataset.split(0.5)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset2), 10)

    def test_summary(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        summary = dataset.summary()
        self.assertEqual(summary.num_instances, 20)
        self.assertEqual(summary.keys['x'].num_unique_values, 20)
        self.assertEqual(summary.keys['ds_id'].num_unique_values, 1)
