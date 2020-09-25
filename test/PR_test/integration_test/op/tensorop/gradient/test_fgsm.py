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

from fastestimator.op.tensorop.gradient import FGSM
from fastestimator.test.unittest_util import is_equal


class TestFGSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_data = tf.Variable([1.0, 2.0, 4.0])
        cls.tf_output = tf.constant([1.01, 2.01, 4])
        cls.torch_data = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        cls.torch_output = torch.tensor([1.01, 2.01, 4])

    def test_tf_input(self):
        fgsm = FGSM(data='x', loss='loss', outputs='y')
        with tf.GradientTape(persistent=True) as tape:
            x = self.tf_data * self.tf_data
            output = fgsm.forward(data=[self.tf_data, x], state={'tape': tape})
        self.assertTrue(np.array_equal(output, self.tf_output))

    def test_torch_input(self):
        fgsm = FGSM(data='x', loss='loss', outputs='y')
        x = self.torch_data * self.torch_data
        output = fgsm.forward(data=[self.torch_data, x], state={'tape': None})
        self.assertTrue(is_equal(output, self.torch_output))
