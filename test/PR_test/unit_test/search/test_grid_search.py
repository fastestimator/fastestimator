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

from fastestimator.search.grid_search import GridSearch


class TestSearch(unittest.TestCase):
    def test_non_dict_params(self):
        with self.assertRaises(AssertionError):
            GridSearch(score_fn=lambda search_idx, x: x, params=False)

    def test_dict_params_wrong_args(self):
        with self.assertRaises(AssertionError):
            GridSearch(score_fn=lambda search_idx, x: x, params={"lr": [1, 2, 3]})

    def test_correct_output(self):
        search = GridSearch(score_fn=lambda search_idx, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
        search.fit()
        self.assertEqual(search.get_best_results(), ({"a": 3, "b": 6, "search_idx": 9}, 9))

    def test_notebook_restart(self):
        search = GridSearch(score_fn=lambda search_idx, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
        search.fit()
        search.fit()
        self.assertEqual(search.search_idx, 9)
