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
import os
import tempfile
import unittest

from fastestimator.search.search import Search


class TestSearch(unittest.TestCase):
    def test_scoring_wo_index(self):
        with self.assertRaises(AssertionError):
            Search(eval_fn=lambda x: x)

    def test_index_increase(self):
        search = Search(eval_fn=lambda search_idx, x: x)
        search.evaluate(x=1)
        search.evaluate(x=2)
        self.assertEqual(search.search_idx, 2)

    def test_caching(self):
        search = Search(eval_fn=lambda search_idx, x: x)
        search.evaluate(x=1)
        search.evaluate(x=1)  # second time should not propagate the index
        self.assertEqual(search.search_idx, 1)

    def test_best_param_max(self):
        search = Search(eval_fn=lambda search_idx, x: x)
        search.evaluate(x=1)
        search.evaluate(x=2)
        self.assertEqual(search.get_best_results(best_mode="max"), {
            "param": {
                "search_idx": 2, "x": 2
            }, "result": {
                "value": 2
            }
        })

    def test_best_param_min(self):
        search = Search(eval_fn=lambda search_idx, x: x)
        search.evaluate(x=1)
        search.evaluate(x=2)
        self.assertEqual(search.get_best_results(best_mode="min"), {
            "param": {
                "search_idx": 1, "x": 1
            }, "result": {
                "value": 1
            }
        })

    def test_get_state(self):
        search = Search(eval_fn=lambda search_idx, x: x)
        search.evaluate(x=1)
        search.evaluate(x=2)
        search.evaluate(x=1)
        search.evaluate(x=5)
        state = search._get_state()
        self.assertEqual(len(state["search_summary"]), 3)

    def test_save(self):
        search = Search(eval_fn=lambda search_idx, x: x, name="unicorn")
        save_dir = tempfile.mkdtemp()
        search.save(save_dir=save_dir)
        self.assertTrue(os.path.exists(os.path.join(save_dir, "unicorn.json")))

    def test_load(self):
        search = Search(eval_fn=lambda search_idx, x: x, name="ultraman")
        search.evaluate(x=1)
        search.evaluate(x=2)
        save_dir = tempfile.mkdtemp()
        search.save(save_dir=save_dir)
        # now to load
        search2 = Search(eval_fn=lambda search_idx, x: x, name="ultraman")
        search2.load(load_dir=save_dir)
        search2.evaluate(x=1)
        self.assertEqual(search2.search_idx, 2)

    def test_load_empty_dir(self):
        search = Search(eval_fn=lambda search_idx, x: x, name="pikachu")
        with self.assertRaises(ValueError):
            search.load(tempfile.mkdtemp())

    def test_load_empty_dir_not_exists_ok(self):
        search = Search(eval_fn=lambda search_idx, x: x, name="pikachu")
        search.load(tempfile.mkdtemp(), not_exist_ok=True)

    def test_eval_fn_single_key(self):
        search = Search(eval_fn=lambda search_idx, x: {"val": x})
        search.evaluate(x=1)
        search.evaluate(x=2)
        self.assertEqual(search.get_best_results(best_mode="min"), {
            "param": {
                "search_idx": 1, "x": 1
            }, "result": {
                "val": 1
            }
        })

    def test_eval_fn_multi_key(self):
        search = Search(eval_fn=lambda search_idx, x: {"val": x, "val2": -x})
        search.evaluate(x=1)
        search.evaluate(x=2)
        self.assertEqual(search.get_best_results(best_mode="min", optimize_field="val"), {
            "param": {
                "search_idx": 1, "x": 1
            }, "result": {
                "val": 1, "val2": -1
            }
        })
