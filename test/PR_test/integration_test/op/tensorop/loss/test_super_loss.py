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
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.op.tensorop.loss import CrossEntropy, Hinge, SuperLoss
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch


class TestSuperLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # torch binary ce
        cls.torch_true_binary = torch.tensor([[1], [0], [1], [0]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.torch_pred_binary = torch.tensor([[0.9], [0.3], [0.8],
                                              [0.1]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        # categorical ce
        cls.tf_true_cat = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cls.tf_pred_cat = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.torch_true_cat = torch.tensor([[0, 1, 0], [1, 0, 0],
                                           [0, 0, 1]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.torch_pred_cat = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05],
                                           [0.1, 0.2, 0.7]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        # sparse categorical ce
        cls.tf_true_sparse = tf.constant([[0], [1], [0]])
        cls.tf_pred_sparse = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.state = {'warmup': False, 'epoch': 1, 'mode': 'train'}

    @staticmethod
    @tf.function
    def do_forward(op, data, state):
        return op.forward(data, state)

    def test_tf_superloss_categorical_ce(self):
        sl = SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
        sl.build(framework="tf", device=None)
        output = sl.forward(data=[self.tf_pred_cat, self.tf_true_cat], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.0026386082))

    def test_tf_static_superloss_categorical_ce(self):
        sl = SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
        sl.build(framework="tf", device=None)
        output = self.do_forward(sl, data=[self.tf_pred_cat, self.tf_true_cat], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.0026386082))

    def test_tf_superloss_sparse_categorical_ce(self):
        sl = SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
        sl.build(framework="tf", device=None)
        output = sl.forward(data=[self.tf_pred_sparse, self.tf_true_sparse], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.024740249))

    def test_tf_static_superloss_sparse_categorical_ce(self):
        sl = SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
        sl.build(framework="tf", device=None)
        output = self.do_forward(sl, data=[self.tf_pred_sparse, self.tf_true_sparse], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.024740249))

    def test_torch_superloss_binary_ce(self):
        sl = SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
        sl.build(framework="torch", device="cuda:0" if torch.cuda.is_available() else "cpu")
        output = sl.forward(data=[self.torch_pred_binary, self.torch_true_binary], state=self.state)
        self.assertTrue(np.allclose(output.detach().to("cpu").numpy(), -0.0026238672))

    def test_tf_superloss_hinge(self):
        true = tf.constant([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                            [1.0, -1.0, -1.0, -1.0]])
        sl = SuperLoss(Hinge(inputs=('x1', 'x2'), outputs='x'))
        sl.build('tf', None)
        output = sl.forward(data=[pred, true], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.072016776))

    def test_tf_static_superloss_hinge(self):
        true = tf.constant([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1], [1, -1, -1, -1]])
        pred = tf.constant([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                            [1.0, -1.0, -1.0, -1.0]])
        sl = SuperLoss(Hinge(inputs=('x1', 'x2'), outputs='x'))
        sl.build('tf', None)
        output = self.do_forward(sl, data=[pred, true], state=self.state)
        self.assertTrue(np.allclose(output.numpy(), -0.072016776))

    def test_torch_superloss_hinge(self):
        true = torch.tensor([[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, -1],
                             [1, -1, -1, -1]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        pred = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.1, -0.2, 0.0, -0.7], [0.0, 0.15, 0.8, 0.05],
                             [1.0, -1.0, -1.0, -1.0]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
        sl = SuperLoss(Hinge(inputs=('x1', 'x2'), outputs='x'))
        sl.build('torch', "cuda:0" if torch.cuda.is_available() else "cpu")
        output = sl.forward(data=[pred, true], state=self.state)
        self.assertTrue(np.allclose(output.to("cpu").numpy(), -0.072016776))

    def test_save_and_load_state_tf(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(ops=[
                ModelOp(model=model, inputs="x_out", outputs="y_pred"),
                SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
            ])
            return system

        system = instantiate_system()

        # make some changes
        system.network.ops[1].forward(data=[self.tf_pred_cat, self.tf_true_cat], state=self.state)

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # re-instantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        loaded_op = system.network.ops[1]
        with self.subTest("Initialization Flags"):
            self.assertEqual(loaded_op.initialized['train'].numpy(), True)
            self.assertEqual(loaded_op.initialized['eval'].numpy(), False)
        with self.subTest("Mean Values"):
            self.assertTrue(np.allclose(loaded_op.tau['train'].numpy(), 0.22839302))
            self.assertTrue(np.allclose(loaded_op.tau['eval'].numpy(), 0.0))

    def test_save_and_load_state_torch(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(ops=[
                ModelOp(model=model, inputs="x_out", outputs="y_pred"),
                SuperLoss(CrossEntropy(inputs=['y_pred', 'y'], outputs='ce'))
            ])
            return system

        system = instantiate_system()
        # make some changes
        system.network.ops[1].forward(data=[self.torch_pred_cat, self.torch_true_cat], state=self.state)

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # re-instantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)

        loaded_op = system.network.ops[1]
        with self.subTest("Initialization Flags"):
            self.assertEqual(loaded_op.initialized['train'].to("cpu").numpy(), True)
            self.assertEqual(loaded_op.initialized['eval'].to("cpu").numpy(), False)
        with self.subTest("Mean Values"):
            self.assertTrue(np.allclose(loaded_op.tau['train'].to("cpu").numpy(), 0.22839302))
            self.assertTrue(np.allclose(loaded_op.tau['eval'].to("cpu").numpy(), 0.0))
