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
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from fastestimator.dataset.dataloader import decollate_batch


class TestDecollateBatch(unittest.TestCase):

    def test_basic_decollate(self):
        instances = [{'a': 0, 'b': np.zeros((28, 28, 3))},
                     {'a': 1, 'b': np.ones((28, 28, 3))},
                     {'a': 2, 'b': np.zeros((28, 28, 3)) - 1}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))

    def test_decollate_list(self):
        instances = [{'a': 0, 'b': [0, 0, 1, 0, 1]},
                     {'a': 1, 'b': [2, 0, 4, 0, 0]},
                     {'a': 2, 'b': [0, 0, 1, 1, 0]}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))

    def test_nested_dict(self):
        instances = [{'a': {'b': [0, 0, 1, 0, 1], 'c': np.zeros((28, 28, 3))}},
                     {'a': {'b': [2, 0, 4, 0, 0], 'c': np.ones((28, 28, 3))}},
                     {'a': {'b': [3, 1, 4, 0, 0], 'c': np.zeros((28, 28, 3)) - 1}}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))

    def test_strings(self):
        instances = [{'a': 0, 'b': "woobop"},
                     {'a': 1, 'b': "with a space"},
                     {'a': 2, 'b': "?"}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))

    def test_named_tuple(self):
        Point = namedtuple('Point', ['x', 'y'])
        instances = [{'a': 0, 'b': Point(5, 4)},
                     {'a': 1, 'b': Point(7, 3)},
                     {'a': 2, 'b': Point(2, 0)}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))

    def test_named_tuple_str(self):
        Point = namedtuple('Point', ['x', 'y'])
        instances = [{'a': 0, 'b': Point("str", "strrr")},
                     {'a': 1, 'b': Point("?", "a")},
                     {'a': 2, 'b': Point("bb", "")}]
        batch = default_collate(instances)

        decollated = decollate_batch(batch)
        self.assertTrue(nested_equal(decollated, instances))

        re_batched = default_collate(decollated)
        self.assertTrue(nested_equal(re_batched, batch))


def nested_equal(a, b) -> bool:
    if isinstance(a, dict):
        if not isinstance(b, dict):
            print(f"{type(a)} != {type(b)}")
            return False
        if set(a.keys()) != set(b.keys()):
            print(f"{set(a.keys())} != {set(a.keys())}")
            return False
        for key in a:
            if not nested_equal(a[key], b[key]):
                print(f"a[{key}] != b[{key}]")
                return False
        return True
    if isinstance(a, (list, set, tuple)):
        if not isinstance(b, a.__class__):
            print(f"{type(a)} != {type(b)}")
            return False
        if len(a) != len(b):
            print(f"len(a) {len(a)} != len(b) {len(b)}")
            return False
        for idx, (e_a, e_b) in enumerate(zip(a, b)):
            if not nested_equal(e_a, e_b):
                print(f"idx {idx} not equal in list")
                return False
        return True
    if isinstance(a, (np.ndarray, torch.Tensor)):
        if not isinstance(b, a.__class__):
            print(f"{type(a)} != {type(b)}")
            return False
        if not np.allclose(a, b):
            print(f"{a} != {b}")
            return False
        return True
    if a == b:
        return True
    print(f"{a} != {b}")
    return False
