# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import shutil
import tempfile
import unittest

from fastestimator.search.grid_search import GridSearch
from fastestimator.search.visualize import visualize_search


class TestVisualize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_path = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.save_path)

    def test_1np1nr_smoke(self):
        def search(search_idx, x1):
            return {'y1': 0}

        name = '1np1nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_1np2nr_smoke(self):
        def search(search_idx, x1):
            return {'y1': 0, 'y2': 1}

        name = '1np2nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_1np3nr_smoke(self):
        def search(search_idx, x1):
            return {'y1': 0, 'y2': 1, 'y3': 2}

        name = '1np3nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_2np1nr_smoke(self):
        def search(search_idx, x1, x2):
            return {'y1': 0}

        name = '2np1nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_2np2nr_smoke(self):
        def search(search_idx, x1, x2):
            return {'y1': 0, 'y2': 1}

        name = '2np2nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_2np3nr_smoke(self):
        def search(search_idx, x1, x2):
            return {'y1': 0, 'y2': 1, 'y3': 2}

        name = '2np3nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_3np1nr_smoke(self):
        def search(search_idx, x1, x2, x3):
            return {'y1': 0}

        name = '3np1nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3], 'x3': [1.4, 2.7]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_3np2nr_smoke(self):
        def search(search_idx, x1, x2, x3):
            return {'y1': 0, 'y2': 1}

        name = '3np2nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3], 'x3': [1.4, 2.7]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_3np3nr_smoke(self):
        def search(search_idx, x1, x2, x3):
            return {'y1': 0, 'y2': 1, 'y3': 2}

        name = '3np3nr'
        s = GridSearch(eval_fn=search, params={'x1': [0, 1, 2, 3], 'x2': [2, 3], 'x3': [1.4, 2.7]}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_2cp1nr_smoke(self):
        def search(search_idx, x1, x2):
            return {'y1': 0}

        name = '2cp1nr'
        s = GridSearch(eval_fn=search, params={'x1': ['adam', 'sgd'], 'x2': ['linear', 'bicubic']}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))

    def test_2cp1cr_smoke(self):
        def search(search_idx, x1, x2):
            return {'y1': f"{x1}+{x2}"}

        name = '2cp1cr'
        s = GridSearch(eval_fn=search, params={'x1': ['adam', 'sgd'], 'x2': ['linear', 'bicubic']}, name=name)
        s.fit()
        visualize_search(search=s, save_path=os.path.join(self.save_path, name))
        self.assertTrue(os.path.exists(os.path.join(self.save_path, f"{name}.html")))
