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
import torch

from fastestimator.op.tensorop.gradient import GradientOp
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model


class TestGradientOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.torch_data = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
        cls.torch_output = [torch.tensor([2, 4, 8], dtype=torch.float32)]
        cls.tf_model = one_layer_tf_model()
        cls.torch_model = OneLayerTorchModel()

    def test_torch_input(self):
        gradient = GradientOp(inputs='x', finals='x', outputs='y')
        gradient.build("torch")
        x = self.torch_data * self.torch_data
        output = gradient.forward(data=[self.torch_data, x], state={'tape': None})
        self.assertTrue(is_equal(output, self.torch_output))
