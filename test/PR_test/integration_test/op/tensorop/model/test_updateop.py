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
from copy import deepcopy

import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.op.tensorop.model import UpdateOp
from fastestimator.test.unittest_util import MultiLayerTorchModel, is_equal, one_layer_tf_model


class TestUpdateOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = {'mode': 'train', 'epoch': 1, 'warmup': False, "deferred": {}, "scaler": None}
        cls.tf_input_data = tf.Variable([[2.0, 1.5, 1.0], [1.0, -1.0, -0.5]])
        cls.torch_input_data = torch.tensor([[1.0, 1.0, 1.0, -0.5], [0.5, 1.0, -1.0, -0.5]],
                                            dtype=torch.float32,
                                            requires_grad=True)
        cls.torch_y = torch.tensor([[5], [7]], dtype=torch.float32)
        cls.tf_y = tf.constant([[-6], [1]])

    def test_tf_input(self):
        model = fe.build(one_layer_tf_model, optimizer_fn="adam")
        op = UpdateOp(model=model, loss_name='loss')
        weights_before = model.layers[1].get_weights()
        with tf.GradientTape(persistent=True) as tape:
            self.state['tape'] = tape
            pred = fe.backend.feed_forward(model, self.tf_input_data)
            loss = fe.backend.mean_squared_error(y_pred=pred, y_true=self.tf_y)
            op.forward(data=loss, state=self.state)
        weights_after = model.layers[1].get_weights()
        self.assertFalse(is_equal(weights_before, weights_after))

    def test_torch_input(self):
        model = fe.build(model_fn=MultiLayerTorchModel, optimizer_fn="adam")
        weights_before = deepcopy(model.fc1.weight.data.numpy())
        op = UpdateOp(model=model, loss_name='loss')
        pred = fe.backend.feed_forward(model, self.torch_input_data)
        loss = fe.backend.mean_squared_error(y_pred=pred, y_true=self.torch_y)
        op.forward(data=loss, state=self.state)
        weights_after = model.fc1.weight.data.numpy()
        self.assertFalse(is_equal(weights_before, weights_after))
