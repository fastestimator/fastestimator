# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.op.tensorop.loss import L1_Loss


class Test_L1_Loss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TensorFlow
        cls.tf_true = tf.constant([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        cls.tf_pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05],
                                   [1.0, 0.0, 0.0, 0.0]])
        # PyTorch
        cls.torch_true = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        cls.torch_pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, 0.2, 0.0, 0.7], [0.0, 0.15, 0.8, 0.05],
                                       [1.0, 0.0, 0.0, 0.0]])
        # State Definition
        cls.state = {'warmup': False, 'epoch': 1, 'mode': 'train'}

    def test_L1_tf(self):
        l1 = L1_Loss(inputs='x', outputs='x')
        output = l1.forward(data=[self.tf_pred, self.tf_true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.081250004))

    def test_L1_torch(self):
        l1 = L1_Loss(inputs='x', outputs='x')
        output = l1.forward(data=[self.torch_pred, self.torch_true], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.081250004))

    def test_Smooth_L1_tf(self):
        smooth_l1 = L1_Loss(inputs='x', outputs='x', loss_type='Smooth', beta=0.65)
        output = smooth_l1.forward(data=[self.tf_pred, self.tf_true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.011057693))

    def test_Smooth_L1_torch(self):
        smooth_l1 = L1_Loss(inputs='x', outputs='x', loss_type='Smooth', beta=0.65)
        output = smooth_l1.forward(data=[self.torch_pred, self.torch_true], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.011057693))

    def test_Huber_tf(self):
        Huber = L1_Loss(inputs='x', outputs='x', loss_type='Huber', beta=0.65)
        output = Huber.forward(data=[self.tf_pred, self.tf_true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.0071875006))

    def test_Huber_torch(self):
        Huber = L1_Loss(inputs='x', outputs='x', loss_type='Huber', beta=0.65)
        output = Huber.forward(data=[self.torch_pred, self.torch_true], state={})
        self.assertTrue(np.allclose(output.detach().numpy(), 0.0071875006))
