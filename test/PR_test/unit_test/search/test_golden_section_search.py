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

from fastestimator.search.golden_section import GoldenSection


class TestSearch(unittest.TestCase):
    def test_reversed_x_max_x_min(self):
        with self.assertRaises(AssertionError):
            GoldenSection(eval_fn=lambda search_idx, n: (n - 3)**2, x_min=6, x_max=0, max_iter=10, best_mode="min")

    def test_invalid_eval_fn(self):
        with self.assertRaises(AssertionError):
            GoldenSection(eval_fn=lambda search_idx, n, q: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min")

    def test_integer_results(self):
        search = GoldenSection(eval_fn=lambda search_idx, n: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min")
        search.fit()
        self.assertEqual(search.get_best_results()["param"]["n"], 3)

    def test_float_results(self):
        search = GoldenSection(
            eval_fn=lambda search_idx, n: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min", integer=False)
        search.fit()
        answer = 3.0
        result = search.get_best_results()["param"]["n"]
        self.assertTrue(abs(result - answer) < 0.01)
