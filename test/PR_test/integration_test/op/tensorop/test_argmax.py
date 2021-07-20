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

from fastestimator.op.tensorop.argmax import Argmax
from fastestimator.test.unittest_util import is_equal


class TestArgmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.constant([1., 2., 4.])
        cls.tf_output = [tf.convert_to_tensor(2, dtype=tf.int64)]
        cls.torch_data = torch.tensor([1., 2., 4.], requires_grad=True)
        cls.torch_output = [torch.tensor(2, dtype=torch.long)]

    def test_tf_input(self):
        argmax = Argmax(inputs='x', outputs='x')
        output = argmax.forward(data=[self.tf_data], state={})
        self.assertTrue(is_equal(output, self.tf_output))

    def test_torch_input(self):
        argmax = Argmax(inputs='x', outputs='x')
        output = argmax.forward(data=[self.torch_data], state={})
        self.assertTrue(is_equal(output, self.torch_output))
