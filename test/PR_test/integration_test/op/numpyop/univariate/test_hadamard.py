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

from fastestimator.op.numpyop.univariate.hadamard import Hadamard
from fastestimator.test.unittest_util import is_equal


class TestHadamard(unittest.TestCase):
    def test_4class(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=4)
        output = tohadamard.forward(data=[np.array([3.0, 2.0, 2.0, 2.0, 0.0])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                np.array([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                          [-1., 1., 1., 1.]])))

    def test_4class_single_input(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=4)
        output = tohadamard.forward(data=[0], state={})[0]
        self.assertTrue(is_equal(output, np.array([-1., 1., 1., 1.])))

    def test_10class_16code(self):
        tohadamard = Hadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        output = tohadamard.forward(data=[np.array([[0], [1], [9], [0], [4]])], state={})[0]
        self.assertTrue(
            is_equal(
                output,
                np.array([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                           1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                 -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                        1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                          [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])))
