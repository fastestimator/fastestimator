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

from fastestimator.op.numpyop.univariate import WordtoId
from fastestimator.test.unittest_util import is_equal


class TestWordToId(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_dict = {'a': 0, 'b': 11, 'test': 90, 'op': 25, 'c': 100, 'id': 10, 'word': 55, 'to': 5}
        cls.single_input = [['a', 'b', 'test', 'op']]
        cls.single_output = [np.array([0, 11, 90, 25])]
        cls.multi_input = [['test', 'op', 'c'], ['word', 'to', 'id']]
        cls.multi_output = [np.array([90, 25, 100]), np.array([55, 5, 10])]

    def mapping_func(self, seq):
        seq_ids = []
        for token in seq:
            if token in self.map_dict:
                seq_ids.append(self.map_dict[token])
            else:
                seq_ids.append(-1)
        return seq_ids

    def test_single_input(self):
        op = WordtoId(inputs='x', outputs='x', mapping=self.map_dict)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_single_input_mapping_function(self):
        op = WordtoId(inputs='x', outputs='x', mapping=self.mapping_func)
        data = op.forward(data=self.single_input, state={})
        self.assertTrue(is_equal(data, self.single_output))

    def test_multi_input(self):
        op = WordtoId(inputs='x', outputs='x', mapping=self.map_dict)
        data = op.forward(data=self.multi_input, state={})
        self.assertTrue(is_equal(data, self.multi_output))
