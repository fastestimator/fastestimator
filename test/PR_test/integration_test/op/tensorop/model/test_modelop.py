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

import fastestimator as fe
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import OneLayerTorchModel, is_equal, one_layer_tf_model


class TestModelOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = {'mode': 'train', 'epoch': 1}
        cls.tf_input_data = tf.constant([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        cls.torch_input_data = torch.Tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        cls.output = np.array([[6.0], [-2.5]])

    def test_tf_input(self):
        model = fe.build(model_fn=one_layer_tf_model, optimizer_fn="adam")
        op = ModelOp(inputs='x', outputs='x', model=model)
        output = op.forward(data=self.tf_input_data, state=self.state)
        self.assertTrue(is_equal(output.numpy(), self.output))

    def test_torch_input(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        self.torch_input_data = self.torch_input_data.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op = ModelOp(inputs='x', outputs='x', model=model)
        output = op.forward(data=self.torch_input_data, state=self.state)
        output = output.to("cpu")
        self.assertTrue(is_equal(output.detach().numpy(), self.output))
