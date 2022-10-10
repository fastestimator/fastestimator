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
import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import Onehot


class TestOnehot(unittest.TestCase):
    def test_one_d(self):
        single_input = [[1], [2], [3], [3]]
        single_output = [np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])]
        op = Onehot(inputs='x', outputs='x', num_classes=4)
        data = op.forward(data=single_input, state={})
        np.testing.assert_array_equal(data, single_output)

    def test_int_input(self):
        int_input = [2]
        int_output = [np.array([0., 0., 1., 0.])]
        op = Onehot(inputs='x', outputs='x', num_classes=4)
        data = op.forward(data=int_input, state={})
        np.testing.assert_array_equal(data, int_output)

    def test_label_smoothing(self):
        label_smoothing_input = [2]
        label_smoothing_output = [np.array([0.025, 0.025, 0.925, 0.025])]
        op = Onehot(inputs='x', outputs='x', num_classes=4, label_smoothing=0.1)
        data = op.forward(data=label_smoothing_input, state={})
        np.testing.assert_array_equal(data, label_smoothing_output)

    def test_two_d(self):
        two_d_input = [np.ones((2, 2)).astype(np.int8)]
        two_d_input[0][1, 1] = 0
        two_d_output = np.array([[[0., 1., 0., 0.], [0., 1., 0., 0.]], [[0., 1., 0., 0.], [1., 0., 0., 0.]]])
        op = Onehot(inputs='x', outputs='x', num_classes=4)
        data = op.forward(data=two_d_input, state={})
        np.testing.assert_array_equal(data[0], two_d_output)
