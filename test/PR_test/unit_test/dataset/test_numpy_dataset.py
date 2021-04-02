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

import numpy as np
import tensorflow as tf

import fastestimator as fe


class TestNumpyDataset(unittest.TestCase):
    def test_dataset(self):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})

        self.assertEqual(len(train_data), 60000)

    def test_single_frac_split(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2 = ds1.split(0.3)
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 70)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 30)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))

    def test_single_frac_split_seed(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2 = ds1.split(0.3, seed=42)
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 70)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 30)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))

        ds3 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds4 = ds3.split(0.3, seed=42)
        with self.subTest("Both source datasets should be equivalent"):
            self.assertEqual(ds1["idx"], ds3["idx"])
        with self.subTest("Both new datasets should be equivalent"):
            self.assertEqual(ds2["idx"], ds4["idx"])

    def test_single_frac_split_stratify(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2 = ds1.split(0.4, stratify="clz")
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 60)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 40)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
        with self.subTest("Source class balance maintained"):
            self.assertEqual(ds1["clz"].count(0), 15)
            self.assertEqual(ds1["clz"].count(1), 15)
            self.assertEqual(ds1["clz"].count(2), 15)
            self.assertEqual(ds1["clz"].count(3), 15)
        with self.subTest("New class balance maintained"):
            self.assertEqual(ds2["clz"].count(0), 10)
            self.assertEqual(ds2["clz"].count(1), 10)
            self.assertEqual(ds2["clz"].count(2), 10)
            self.assertEqual(ds2["clz"].count(3), 10)

    def test_single_frac_split_seed_stratify(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2 = ds1.split(0.4, seed=17, stratify="clz")
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 60)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 40)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
        with self.subTest("Source class balance maintained"):
            self.assertEqual(ds1["clz"].count(0), 15)
            self.assertEqual(ds1["clz"].count(1), 15)
            self.assertEqual(ds1["clz"].count(2), 15)
            self.assertEqual(ds1["clz"].count(3), 15)
        with self.subTest("New class balance maintained"):
            self.assertEqual(ds2["clz"].count(0), 10)
            self.assertEqual(ds2["clz"].count(1), 10)
            self.assertEqual(ds2["clz"].count(2), 10)
            self.assertEqual(ds2["clz"].count(3), 10)

        ds3 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds4 = ds3.split(0.4, seed=17, stratify="clz")
        with self.subTest("Both source datasets should be equivalent"):
            self.assertEqual(ds1["idx"], ds3["idx"])
        with self.subTest("Both new datasets should be equivalent"):
            self.assertEqual(ds2["idx"], ds4["idx"])

    def test_multi_frac_split(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2, ds3 = ds1.split(0.3, 0.4)
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 30)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 30)
            self.assertEqual(len(ds3), 40)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
            self.assertEqual(set(), set(ds2["idx"]) & set(ds3["idx"]))
            self.assertEqual(set(), set(ds1["idx"]) & set(ds3["idx"]))

    def test_multi_frac_split_seed(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2, ds3 = ds1.split(0.3, 0.4, seed=42)
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 30)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 30)
            self.assertEqual(len(ds3), 40)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
            self.assertEqual(set(), set(ds2["idx"]) & set(ds3["idx"]))
            self.assertEqual(set(), set(ds1["idx"]) & set(ds3["idx"]))

        ds4 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds5, ds6 = ds4.split(0.3, 0.4, seed=42)
        with self.subTest("Both source datasets should be equivalent"):
            self.assertEqual(ds1["idx"], ds4["idx"])
        with self.subTest("New datasets should be equivalent"):
            self.assertEqual(ds2["idx"], ds5["idx"])
            self.assertEqual(ds3["idx"], ds6["idx"])

    def test_multi_frac_split_stratify(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2, ds3 = ds1.split(0.4, 0.2, stratify="clz")
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 40)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 40)
            self.assertEqual(len(ds3), 20)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
            self.assertEqual(set(), set(ds2["idx"]) & set(ds3["idx"]))
            self.assertEqual(set(), set(ds1["idx"]) & set(ds3["idx"]))
        with self.subTest("Source class balance maintained"):
            self.assertEqual(ds1["clz"].count(0), 10)
            self.assertEqual(ds1["clz"].count(1), 10)
            self.assertEqual(ds1["clz"].count(2), 10)
            self.assertEqual(ds1["clz"].count(3), 10)
        with self.subTest("New class balance maintained"):
            self.assertEqual(ds2["clz"].count(0), 10)
            self.assertEqual(ds2["clz"].count(1), 10)
            self.assertEqual(ds2["clz"].count(2), 10)
            self.assertEqual(ds2["clz"].count(3), 10)
            self.assertEqual(ds3["clz"].count(0), 5)
            self.assertEqual(ds3["clz"].count(1), 5)
            self.assertEqual(ds3["clz"].count(2), 5)
            self.assertEqual(ds3["clz"].count(3), 5)

    def test_multi_frac_split_seed_stratify(self):
        ds1 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds2, ds3 = ds1.split(0.4, 0.2, seed=17, stratify="clz")
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 40)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 40)
            self.assertEqual(len(ds3), 20)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
            self.assertEqual(set(), set(ds2["idx"]) & set(ds3["idx"]))
            self.assertEqual(set(), set(ds1["idx"]) & set(ds3["idx"]))
        with self.subTest("Source class balance maintained"):
            self.assertEqual(ds1["clz"].count(0), 10)
            self.assertEqual(ds1["clz"].count(1), 10)
            self.assertEqual(ds1["clz"].count(2), 10)
            self.assertEqual(ds1["clz"].count(3), 10)
        with self.subTest("New class balance maintained"):
            self.assertEqual(ds2["clz"].count(0), 10)
            self.assertEqual(ds2["clz"].count(1), 10)
            self.assertEqual(ds2["clz"].count(2), 10)
            self.assertEqual(ds2["clz"].count(3), 10)
            self.assertEqual(ds3["clz"].count(0), 5)
            self.assertEqual(ds3["clz"].count(1), 5)
            self.assertEqual(ds3["clz"].count(2), 5)
            self.assertEqual(ds3["clz"].count(3), 5)

        ds4 = fe.dataset.NumpyDataset(
            {"idx": np.array([i for i in range(100)]), "clz": np.array([i % 4 for i in range(100)])})
        ds5, ds6 = ds4.split(0.4, 0.2, seed=17, stratify="clz")
        with self.subTest("Both source datasets should be equivalent"):
            self.assertEqual(ds1["idx"], ds4["idx"])
        with self.subTest("New datasets should be equivalent"):
            self.assertEqual(ds2["idx"], ds5["idx"])
            self.assertEqual(ds3["idx"], ds6["idx"])

    def test_heavily_imbalanced_stratify_split(self):
        idx_array = np.array([i for i in range(100)])
        clz_array = np.array(
            [0 if i < 10 else 1 if i < 40 else 2 if i < 77 else 3 if i < 84 else 4 if i < 97 else 5 if i < 99 else 6 for
             i in range(100)])
        # 0: 10%, 1: 30%, 2: 37%, 3: 7%, 4: 13%, 5: 2%, 6: 1%
        ds1 = fe.dataset.NumpyDataset(
            {"idx": idx_array, "clz": clz_array})
        ds2, ds3 = ds1.split(0.3, 0.25, stratify="clz")
        with self.subTest("Source Dataset Size"):
            self.assertEqual(len(ds1), 45)
        with self.subTest("New Dataset Size"):
            self.assertEqual(len(ds2), 30)
            self.assertEqual(len(ds3), 25)
        with self.subTest("Disjoint datasets"):
            self.assertEqual(set(), set(ds1["idx"]) & set(ds2["idx"]))
            self.assertEqual(set(), set(ds2["idx"]) & set(ds3["idx"]))
            self.assertEqual(set(), set(ds1["idx"]) & set(ds3["idx"]))
        with self.subTest("DS1 class balance maintained"):
            self.assertEqual(ds1["clz"].count(0), 5)  # 11.11%
            self.assertEqual(ds1["clz"].count(1), 13)  # 28.88%
            self.assertEqual(ds1["clz"].count(2), 18)  # 40%
            self.assertEqual(ds1["clz"].count(3), 3)  # 6.66%
            self.assertEqual(ds1["clz"].count(4), 6)  # 13.33%
            self.assertEqual(ds1["clz"].count(5), 0)  # 0%
            self.assertEqual(ds1["clz"].count(6), 0)  # 0%
        with self.subTest("DS2 class balance maintained"):
            self.assertEqual(ds2["clz"].count(0), 3)  # 10%
            self.assertEqual(ds2["clz"].count(1), 9)  # 30%
            self.assertEqual(ds2["clz"].count(2), 10)  # 33.33%
            self.assertEqual(ds2["clz"].count(3), 2)  # 6.66%
            self.assertEqual(ds2["clz"].count(4), 4)  # 13.33%
            self.assertEqual(ds2["clz"].count(5), 1)  # 3.33%
            self.assertEqual(ds2["clz"].count(6), 1)  # 3.33%
        with self.subTest("DS3 class balance maintained"):
            self.assertEqual(ds3["clz"].count(0), 2)  # 8%
            self.assertEqual(ds3["clz"].count(1), 8)  # 32%
            self.assertEqual(ds3["clz"].count(2), 9)  # 36%
            self.assertEqual(ds3["clz"].count(3), 2)  # 8%
            self.assertEqual(ds3["clz"].count(4), 3)  # 12%
            self.assertEqual(ds3["clz"].count(5), 1)  # 4%
            self.assertEqual(ds3["clz"].count(6), 0)  # 0%
