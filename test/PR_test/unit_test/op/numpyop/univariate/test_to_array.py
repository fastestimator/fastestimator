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

from fastestimator.op.numpyop.univariate import ToArray
from fastestimator.test.unittest_util import is_equal


class TestToArray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = [1, 2, 3]
        cls.output = [np.array(1), np.array(2), np.array(3)]

    def test_output_values(self):
        op = ToArray(inputs='x', outputs='x')
        data = op.forward(data=self.input, state={})
        self.assertTrue(is_equal(data, self.output))
