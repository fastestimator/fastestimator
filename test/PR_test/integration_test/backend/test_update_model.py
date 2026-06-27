# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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
from copy import deepcopy

import numpy as np
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import OneLayerTorchModel


class TestUpdateModel(unittest.TestCase):

    def test_torch_model_with_get_gradient(self):
        lr = 0.1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = fe.build(model_fn=OneLayerTorchModel,
                         optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=lr)).to(device)
        init_weights = [deepcopy(x).cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        x = torch.tensor([1.0, 1.0, 1.0]).to(torch.device(device))
        y = fe.backend.feed_forward(model.module if torch.cuda.device_count() > 1 else model, x)

        gradients = fe.backend.get_gradient(target=y, sources=[x for x in model.parameters() if x.requires_grad])
        fe.backend.update_model(model, gradients=gradients)

        gradients = [x.cpu().detach().numpy() for x in gradients]
        new_weights = [x.cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr
            self.assertTrue(np.allclose(new_w_ans, new_w))

    def test_torch_model_with_arbitrary_gradient(self):
        lr = 0.1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = fe.build(model_fn=OneLayerTorchModel,
                         optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=lr)).to(device)
        init_weights = [deepcopy(x).cpu().detach().numpy() for x in model.parameters() if x.requires_grad]
        gradients = [torch.tensor([[1.0, 1.0, 1.0]]).to(torch.device(device))]
        fe.backend.update_model(model, gradients=gradients)

        gradients = [x.cpu().detach().numpy() for x in gradients]
        new_weights = [x.cpu().detach().numpy() for x in model.parameters() if x.requires_grad]

        for init_w, new_w, grad in zip(init_weights, new_weights, gradients):
            new_w_ans = init_w - grad * lr
            self.assertTrue(np.allclose(new_w_ans, new_w))
