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

from fastestimator.summary import Summary, ValWithError, average_summaries


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


class TestSummaryRepr(unittest.TestCase):
    def test_repr_empty(self):
        s = Summary(name='test')
        self.assertIn("test", repr(s))
        self.assertIn("modes=[]", repr(s))

    def test_repr_with_history(self):
        s = Summary(name='exp1')
        s.history['train']['loss'][10] = 0.5
        r = repr(s)
        self.assertIn("exp1", r)
        self.assertIn("train", r)
        self.assertIn("loss", r)


class TestSummaryLen(unittest.TestCase):
    def test_empty(self):
        s = Summary(name='test')
        self.assertEqual(len(s), 0)

    def test_with_data(self):
        s = Summary(name='test')
        s.history['train']['loss'] = {1: 0.9, 2: 0.8, 3: 0.7}
        s.history['eval']['acc'] = {1: 0.5, 2: 0.6}
        self.assertEqual(len(s), 5)


class TestSummaryGetKeys(unittest.TestCase):
    def test_all_keys(self):
        s = Summary(name='test')
        s.history['train']['loss'][1] = 0.5
        s.history['train']['acc'][1] = 0.8
        s.history['eval']['val_loss'][1] = 0.6
        keys = s.get_keys()
        self.assertEqual(keys, ['acc', 'loss', 'val_loss'])

    def test_keys_by_mode(self):
        s = Summary(name='test')
        s.history['train']['loss'][1] = 0.5
        s.history['eval']['val_loss'][1] = 0.6
        self.assertEqual(s.get_keys('train'), ['loss'])
        self.assertEqual(s.get_keys('eval'), ['val_loss'])
        self.assertEqual(s.get_keys('test'), [])


class TestSummaryGetBest(unittest.TestCase):
    def test_best_max(self):
        s = Summary(name='test')
        s.history['eval']['acc'] = {10: 0.5, 20: 0.9, 30: 0.7}
        result = s.get_best('acc', mode='eval', largest=True)
        self.assertEqual(result, (20, 0.9))

    def test_best_min(self):
        s = Summary(name='test')
        s.history['eval']['loss'] = {10: 0.9, 20: 0.3, 30: 0.5}
        result = s.get_best('loss', mode='eval', largest=False)
        self.assertEqual(result, (20, 0.3))

    def test_best_with_val_with_error(self):
        s = Summary(name='test')
        s.history['eval']['acc'] = {10: 0.5, 20: ValWithError(0.7, 0.9, 1.1)}
        result = s.get_best('acc', mode='eval', largest=True)
        self.assertEqual(result[0], 20)
        self.assertIsInstance(result[1], ValWithError)

    def test_best_missing_key(self):
        s = Summary(name='test')
        self.assertIsNone(s.get_best('nonexistent', mode='eval'))

    def test_best_missing_mode(self):
        s = Summary(name='test')
        s.history['train']['acc'] = {10: 0.5}
        self.assertIsNone(s.get_best('acc', mode='eval'))


class TestSummaryToDict(unittest.TestCase):
    def test_to_dict_all(self):
        s = Summary(name='test')
        s.history['train']['loss'] = {1: 0.9, 2: 0.8}
        s.history['eval']['acc'] = {1: 0.5}
        d = s.to_dict()
        self.assertEqual(d['train']['loss'], {1: 0.9, 2: 0.8})
        self.assertEqual(d['eval']['acc'], {1: 0.5})

    def test_to_dict_single_mode(self):
        s = Summary(name='test')
        s.history['train']['loss'] = {1: 0.9}
        s.history['eval']['acc'] = {1: 0.5}
        d = s.to_dict(mode='train')
        self.assertIn('loss', d)
        self.assertNotIn('acc', d)

    def test_to_dict_missing_mode(self):
        s = Summary(name='test')
        d = s.to_dict(mode='test')
        self.assertEqual(d, {})


class TestValWithErrorExtended(unittest.TestCase):
    def test_repr(self):
        v = ValWithError(0.1, 0.5, 0.9)
        self.assertEqual(repr(v), "ValWithError(y_min=0.1, y=0.5, y_max=0.9)")

    def test_hash(self):
        v1 = ValWithError(0.1, 0.5, 0.9)
        v2 = ValWithError(0.1, 0.5, 0.9)
        self.assertEqual(hash(v1), hash(v2))

    def test_ne(self):
        v1 = ValWithError(0.1, 0.5, 0.9)
        v2 = ValWithError(0.2, 0.6, 1.0)
        self.assertNotEqual(v1, v2)

    def test_format(self):
        v = ValWithError(0.123456, 0.5, 0.876543)
        formatted = f"{v:.2f}"
        self.assertEqual(formatted, "(0.12, 0.50, 0.88)")

    def test_in_set(self):
        v1 = ValWithError(0.1, 0.5, 0.9)
        v2 = ValWithError(0.1, 0.5, 0.9)
        s = {v1}
        self.assertIn(v2, s)
