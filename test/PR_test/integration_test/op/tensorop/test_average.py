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
import tensorflow as tf
import torch

from fastestimator.op.tensorop.average import Average
from fastestimator.test.unittest_util import is_equal


class TestAverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = [tf.constant([1., 2., 4.]), tf.constant([1., 4., 5.])]
        cls.tf_output = tf.constant([1., 3., 4.5])
        cls.torch_data = [torch.tensor([1., 2., 4.]), torch.tensor([1., 4., 5.])]
        cls.torch_output = torch.tensor([1., 3., 4.5])

    def test_tf_input(self):
        average = Average(inputs='x', outputs='x')
        output = average.forward(data=self.tf_data, state={})
        self.assertTrue(is_equal(output, self.tf_output))

    def test_torch_input(self):
        average = Average(inputs='x', outputs='x')
        output = average.forward(data=self.torch_data, state={})
        self.assertTrue(is_equal(output, self.torch_output))
