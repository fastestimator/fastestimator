# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.summary.logs.log_plot import _MetricGroup


class TestMetricGroups(unittest.TestCase):
    def test_add(self):
        group = _MetricGroup()
        group.add(exp_id=0, mode='train', ds_id='ds1', values={0: 5, 10: 17})
        group.add(exp_id=0, mode='train', ds_id='ds2', values={0: 7, 8: 19})
        group.add(exp_id=1, mode='eval', ds_id='ds2', values={3: 2.6})

        self.assertTrue(np.array_equal(group[0]['train']['ds1'], np.array([[0, 5], [10, 17]])))
        self.assertTrue(np.array_equal(group[0]['train']['ds2'], np.array([[0, 7], [8, 19]])))
        self.assertTrue(np.array_equal(group[1]['eval']['ds2'], np.array([[3, 2.6]])))

    def test_ndim(self):
        with self.subTest("1D"):
            group = _MetricGroup()
            group.add(exp_id=0, mode='train', ds_id='ds1', values={0: 5})
            group.add(exp_id=0, mode='train', ds_id='ds2', values={8: 19})
            group.add(exp_id=1, mode='eval', ds_id='ds2', values={3: 2.6})
            self.assertEqual(group.ndim(), 1)
        with self.subTest("2D"):
            group = _MetricGroup()
            group.add(exp_id=0, mode='train', ds_id='ds1', values={0: 5})
            group.add(exp_id=0, mode='train', ds_id='ds2', values={0: 7, 8: 19})
            group.add(exp_id=1, mode='eval', ds_id='ds2', values={3: 2.6})
            self.assertEqual(group.ndim(), 2)

    def test_modes(self):
        group = _MetricGroup()
        group.add(exp_id=0, mode='train', ds_id='ds1', values={0: 5, 10: 17})
        group.add(exp_id=0, mode='train', ds_id='ds2', values={0: 7, 8: 19})
        group.add(exp_id=1, mode='train', ds_id='ds1', values={0: 5, 10: 17})
        group.add(exp_id=1, mode='eval', ds_id='ds2', values={3: 2.6})

        with self.subTest("1 Mode"):
            self.assertEquals(group.modes(0), ['train'])
        with self.subTest("2 Modes"):
            self.assertEquals(group.modes(1), ['train', 'eval'])

    def test_ds_ids(self):
        group = _MetricGroup()
        group.add(exp_id=0, mode='train', ds_id='ds1', values={0: 5, 10: 17})
        group.add(exp_id=0, mode='train', ds_id='ds2', values={0: 7, 8: 19})
        group.add(exp_id=1, mode='train', ds_id='ds1', values={0: 5, 10: 17})
        group.add(exp_id=1, mode='eval', ds_id='ds3', values={3: 2.6})

        with self.subTest("All Ids Single Mode"):
            self.assertEquals(group.ds_ids(0), ['ds1', 'ds2'])
        with self.subTest("Mode target present"):
            self.assertEquals(group.ds_ids(0, mode='train'), ['ds1', 'ds2'])
        with self.subTest("Mode target absent"):
            self.assertEquals(group.ds_ids(0, mode='test'), [])
        with self.subTest("All Ids Multi Mode"):
            self.assertEquals(group.ds_ids(1), ['ds1', 'ds3'])
        with self.subTest("Mode target present"):
            self.assertEquals(group.ds_ids(1, mode='train'), ['ds1'])
            self.assertEquals(group.ds_ids(1, mode='eval'), ['ds3'])
        with self.subTest("Mode target absent"):
            self.assertEquals(group.ds_ids(1, mode='test'), [])
