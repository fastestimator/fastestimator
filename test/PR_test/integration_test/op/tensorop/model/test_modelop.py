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

import fastestimator as fe
from fastestimator.backend._get_gradient import get_gradient
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import MultiLayerTorchModel, OneLayerTorchModel, is_equal, multi_layer_tf_model, \
    one_layer_tf_model


class TestModelOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.state = {'mode': 'train', 'epoch': 1, 'tape': None}
        cls.torch_input_data = torch.Tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -0.5]])
        cls.output = np.array([[6.0], [-2.5]])
        # Big inputs
        cls.torch_input_data_big = torch.Tensor([[1.0, 1.0, 1.0, 1.0], [0.6, -0.5, -0.8, 1.2]])
        cls.output_big = np.array([[40.0], [9.4]])
        cls.embedding_output = np.array([[10.0, 15.0], [2.0, 3.7]])

    def test_torch_input(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        self.torch_input_data = self.torch_input_data.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op = ModelOp(inputs='x', outputs='x', model=model)
        op.build(framework='torch', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        output = op.forward(data=self.torch_input_data, state=self.state)
        output = output.to("cpu")
        self.assertTrue(is_equal(output.detach().numpy(), self.output))

    def test_torch_multi_output_str(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn="adam")
        self.torch_input_data_big = self.torch_input_data_big.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op = ModelOp(inputs='x', outputs=['y', 'embedding'], model=model, intermediate_layers='fc1')
        op.build(framework='torch', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        y, embedding = op.forward(data=self.torch_input_data_big, state=self.state)
        y = y.to("cpu")
        embedding = embedding.to('cpu')
        self.assertTrue(np.allclose(y.detach().numpy(), self.output_big, atol=1e-4))
        self.assertTrue(np.allclose(embedding.detach().numpy(), self.embedding_output, atol=1e-4))

    def test_torch_multi_output_int(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn="adam")
        self.torch_input_data_big = self.torch_input_data_big.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op = ModelOp(inputs='x', outputs=['y', 'embedding'], model=model, intermediate_layers=1)
        op.build(framework='torch', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        y, embedding = op.forward(data=self.torch_input_data_big, state=self.state)
        y = y.to("cpu")
        embedding = embedding.to('cpu')
        self.assertTrue(np.allclose(y.detach().numpy(), self.output_big, atol=1e-4))
        self.assertTrue(np.allclose(embedding.detach().numpy(), self.embedding_output, atol=1e-4))

    def test_torch_with_gradients(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=None)
        op = ModelOp(inputs='x', outputs='x', model=model, gradients=True)
        self.torch_input_data = self.torch_input_data.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op.build(framework='torch', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        output = op.forward(data=self.torch_input_data, state=self.state)
        self.assertIsNotNone(output.grad_fn)

    def test_torch_disable_gradients(self):
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn=None)
        op = ModelOp(inputs='x', outputs='x', model=model, gradients=False, trainable=False)
        self.torch_input_data = self.torch_input_data.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        op.build(framework='torch', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        output = op.forward(data=self.torch_input_data, state=self.state)
        self.assertIsNone(output.grad_fn)
