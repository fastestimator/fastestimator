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

from fastestimator.op.tensorop.loss import CrossEntropy


class TestCrossEntropy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # torch binary ce
        cls.torch_true_binary = torch.tensor([[1], [0], [1], [0]])
        cls.torch_pred_binary = torch.tensor([[0.9], [0.3], [0.8], [0.1]])
        cls.torch_binary_weights = {1: 2.0}
        # categorical ce
        cls.torch_true_cat = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cls.torch_pred_cat = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.torch_cat_weights = {1: 2.0, 2: 3.0}
        # sparse categorical ce
        cls.torch_true_sparse = torch.tensor([[0], [1], [0]])
        cls.torch_pred_sparse = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.torch_sparse_weights = {1: 2.0, 2: 3.0}

    def test_categorical_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.torch_pred_cat, self.torch_true_cat], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.22839302))

    def test_categorical_crossentropy_weights(self):
        ce = CrossEntropy(inputs='x', outputs='x', class_weights=self.torch_cat_weights)
        ce.build('torch')
        output = ce.forward(data=[self.torch_pred_cat, self.torch_true_cat], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.54055756))

    def test_sparse_categorical_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.torch_pred_sparse, self.torch_true_sparse], state={})
        self.assertTrue(np.allclose(output.numpy(), 2.5336342))

    def test_sparse_categorical_crossentropy_weights(self):
        ce = CrossEntropy(inputs='x', outputs='x', class_weights=self.torch_sparse_weights)
        ce.build('torch')
        output = ce.forward(data=[self.torch_pred_sparse, self.torch_true_sparse], state={})
        self.assertTrue(np.allclose(output.numpy(), 3.532212))

    def test_binary_crossentropy(self):
        ce = CrossEntropy(inputs='x', outputs='x')
        output = ce.forward(data=[self.torch_pred_binary, self.torch_true_binary], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.1976349))

    def test_binary_crossentropy_weights(self):
        ce = CrossEntropy(inputs='x', outputs='x', class_weights=self.torch_binary_weights)
        ce.build('torch')
        output = ce.forward(data=[self.torch_pred_binary, self.torch_true_binary], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.2797609))
