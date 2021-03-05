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

from fastestimator.summary import average_summaries, Summary, ValWithError


class TestSummary(unittest.TestCase):
    def test_merge(self):
        summary1 = Summary(name='test1')
        summary1.history['train']['acc'] = {50: 0.9}

        summary2 = Summary(name='test2')
        summary2.history['train']['acc'] = {80: 0.8}

        summary1.merge(summary2)

        self.assertEqual(summary1.history['train']['acc'][50], 0.9)
        self.assertEqual(summary1.history['train']['acc'][80], 0.8)


class TestAverageSummaries(unittest.TestCase):
    def test_empty_summaries(self):
        result = average_summaries(name="ex1", summaries=[])
        with self.subTest("Should return a summary instance"):
            self.assertTrue(isinstance(result, Summary))
        with self.subTest("Name should be correct"):
            self.assertEqual("ex1", result.name)

    def test_single_summary(self):
        s1 = Summary('ex2')
        s1.history['train']['acc'][10] = 0.55
        result = average_summaries(name="ex1", summaries=[s1])
        with self.subTest("Should return a summary instance"):
            self.assertTrue(isinstance(result, Summary))
        with self.subTest("Name should be correct"):
            self.assertEqual("ex1", result.name)
        with self.subTest("History should be correct"):
            self.assertEqual(0.55, result.history['train']['acc'][10])

    def test_multi_summaries(self):
        s1 = Summary('s1')
        s2 = Summary('s2')

        s1.history['train']['acc'] = {0: 0.2, 10: 0.4, 20: 0.7, 45: ValWithError(0.8, 0.9, 1.0)}
        s2.history['train']['acc'] = {0: 0.3, 10: 0.4, 20: 0.9, 30: 1.0}

        s1.history['test']['mcc'] = {45: 0.834}

        s1.history['eval']['wombats'] = {5: 4, 10: 9}
        s2.history['eval']['wombats'] = {5: '3 wombats', 10: '7 wombats'}

        s_merge = average_summaries('s', [s1, s2])

        with self.subTest("Should return a summary instance"):
            self.assertTrue(isinstance(s_merge, Summary))
        with self.subTest("Name should be correct"):
            self.assertEqual("s", s_merge.name)
        with self.subTest("Paired datapoints should be matched"):
            self.assertEqual({0, 10, 20, 30, 45}, s_merge.history['train']['acc'].keys())
        with self.subTest("Paired datapoints should have correct mean and std"):
            self.assertEqual(0.25, s_merge.history['train']['acc'][0].y)
            self.assertEqual(0.17929, round(s_merge.history['train']['acc'][0].y_min, 5))
            self.assertEqual(0.32071, round(s_merge.history['train']['acc'][0].y_max, 5))
        with self.subTest("Paired datapoints with zero std should be handled correctly"):
            self.assertEqual(0.4, s_merge.history['train']['acc'][10])
        with self.subTest("Partially paired datapoints should be handled correctly"):
            self.assertEqual(0.9, s_merge.history['train']['acc'][45])
            self.assertEqual(1.0, s_merge.history['train']['acc'][30])
        with self.subTest("Unpaired datapoints should be handled correctly"):
            self.assertEqual(0.834, s_merge.history['test']['mcc'][45])
        with self.subTest("String values should be handled correctly"):
            self.assertEqual(3.5, s_merge.history['eval']['wombats'][5].y)
            self.assertEqual(2.79289, round(s_merge.history['eval']['wombats'][5].y_min, 5))
            self.assertEqual(8.0, s_merge.history['eval']['wombats'][10].y)
