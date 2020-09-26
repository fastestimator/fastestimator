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

import tensorflow as tf
import torch

from fastestimator.op.tensorop.reshape import Reshape
from fastestimator.test.unittest_util import is_equal


class TestReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.constant([[1., 2., 4.], [1., 2., 6.]])
        cls.tf_output = tf.constant([[1., 2., 4., 1., 2., 6.]])
        cls.torch_data = torch.tensor([[1., 2., 4.], [1., 2., 6.]])
        cls.torch_output = torch.tensor([[1., 2., 4., 1., 2., 6.]])

    def test_tf_input(self):
        reshape = Reshape(inputs='x', outputs='x', shape=(1, 6))
        output = reshape.forward(data=[self.tf_data], state={})
        self.assertTrue(is_equal(output[0], self.tf_output))

    def test_torch_input(self):
        reshape = Reshape(inputs='x', outputs='x', shape=(1, 6))
        output = reshape.forward(data=[self.torch_data], state={})
        self.assertTrue(is_equal(output[0], self.torch_output))
