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

import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestCropping2D(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(list(range(100))).view((1, 1, 10, 10))

    def test_cropping_2d_1arg(self):
        op = torch.tensor([[[[33, 34, 35, 36], [43, 44, 45, 46], [53, 54, 55, 56], [63, 64, 65, 66]]]])
        m = fe.layers.pytorch.Cropping2D(3)
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))

    def test_cropping_2d_2arg(self):
        op = torch.tensor([[[[34, 35], [44, 45], [54, 55], [64, 65]]]])
        m = fe.layers.pytorch.Cropping2D((3, 4))
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))

    def test_cropping_2d_tuple(self):
        op = torch.tensor([[[[14, 15], [24, 25], [34, 35], [44, 45], [54, 55]]]])
        m = fe.layers.pytorch.Cropping2D(((1, 4), 4))
        y = m.forward(self.x)
        self.assertTrue(fet.is_equal(y, op))
