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
from fastestimator.op.tensorop.loss import L2Regularizaton
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.architecture.pytorch import LeNet
from fastestimator.trace.metric import Accuracy


class TestL2Regularization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Weight Decay pytorch model
        cls.beta = 0.1
        fe.enable_deterministic(42)
        cls.pytorch_wd = fe.build(model_fn=LeNet, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01, weight_decay=cls.beta))

        # L2 Regularization pytorch model
        fe.enable_deterministic(42)
        cls.pytorch_l2 = fe.build(model_fn=LeNet, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01))

    def test_l2_regularization_pytorch(self):
        fe.enable_deterministic(42)
        pytorch_l2 = fe.build(model_fn=LeNet, optimizer_fn="adam")
        l2 = L2Regularizaton(inputs='x', outputs='x',model=pytorch_l2)
        output = l2.forward(data=torch.tensor([0.0]),state={})
        output = output.detach()
        self.assertTrue(np.allclose(output.numpy(), 0.3984476))

    def test_l2_regularization_tensorflow(self):
        from fastestimator.architecture.tensorflow import LeNet
        fe.enable_deterministic(42)
        tf_l2 = fe.build(model_fn=LeNet, optimizer_fn="adam")
        l2 = L2Regularizaton(inputs='x', outputs='x',model=tf_l2)
        output = l2.forward(data=tf.constant([0.0]),state={})
        self.assertTrue(np.allclose(output.numpy(), 1.1974102))

    def test_pytorch_weight_decay_vs_l2(self):
        # Get Data
        train_data, _ = mnist.load_data()
        t_d = train_data.split(128)
        # Initialize pipeline
        pipeline = fe.Pipeline(train_data=t_d,
                               batch_size=128,
                               ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
        # Define the two pytorch networks
        network_weight_decay = fe.Network(ops=[
            ModelOp(model=self.pytorch_wd, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=self.pytorch_wd, loss_name="ce")
        ])

        network_l2 = fe.Network(ops=[
            ModelOp(model=self.pytorch_l2, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            L2Regularizaton(inputs="ce", outputs="l2", model=self.pytorch_l2, beta=self.beta),
            UpdateOp(model=self.pytorch_l2, loss_name="l2")
        ])

        # defining traces
        traces = [Accuracy(true_key="y", pred_key="y_pred")]

        # Setting up estimators
        estimator_wd = fe.Estimator(pipeline=pipeline,
                                network=network_weight_decay,
                                epochs=1,
                                traces=traces,
                                train_steps_per_epoch=1)

        estimator_l2 = fe.Estimator(pipeline=pipeline,
                                network=network_l2,
                                epochs=1,
                                traces=traces,
                                train_steps_per_epoch=1)
        # Training
        print('********************************Pytorch weight decay training************************************')
        estimator_wd.fit()
        print()
        print('********************************Pytorch L2 Regularization training************************************')
        estimator_l2.fit()
        # testing weights
        count = 0
        for wt, l2 in zip(self.pytorch_wd.parameters(), self.pytorch_l2.parameters()):
            if ((wt-l2).abs()).sum() < torch.tensor(10**-7):
                count += 1
        self.assertTrue(count == 10)

