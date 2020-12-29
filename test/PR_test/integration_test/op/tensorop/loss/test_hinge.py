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

from fastestimator.op.tensorop.loss import Hinge


class TestHinge(unittest.TestCase):
    def test_tf(self):
        true = tf.constant([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                            [1.0, -1.0, -1.0, -1.0]])
        hinge = Hinge(inputs=('x1', 'x2'), outputs='x')
        hinge.build('tf', None)
        output = hinge.forward(data=[pred, true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.7125))

    def test_torch(self):
        true = torch.tensor([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                             [1.0, -1.0, -1.0, -1.0]])
        hinge = Hinge(inputs=('x1', 'x2'), outputs='x')
        hinge.build('torch', torch.device('cpu'))
        output = hinge.forward(data=[pred, true], state={})
        self.assertTrue(np.allclose(output.numpy(), 0.7125))
