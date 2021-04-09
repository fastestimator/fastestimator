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
from collections import deque
from copy import deepcopy

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet as LeNet_torch
from fastestimator.architecture.tensorflow import LeNet as LeNet_tf
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.gradient import GradientOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


class CheckNetworkWeight(fe.trace.Trace):
    def __init__(self, model, grad_key, merge_grad, test_self, lr, framework, activate_epoch=1):
        super().__init__(inputs=grad_key)
        self.model = model
        self.grad_key = grad_key
        self.merge_grad = merge_grad
        self.test_self = test_self
        self.lr = lr
        self.framework = framework
        self.activate_epoch = activate_epoch
        if self.framework == "tf":
            self.previous_weights = [x.numpy() for x in model.trainable_variables]
        else:
            self.previous_weights = [
                deepcopy(x).cpu().detach().numpy() for x in self.model.parameters() if x.requires_grad
            ]
        self.gradients = deque(maxlen=merge_grad)
        self.new_weight = None

    def on_batch_end(self, data):
        if self.system.epoch_idx < self.activate_epoch:
            return
        if self.framework == "tf":
            self.gradients.append([x.numpy() for x in data[self.grad_key]])
            self.new_weight = [x.numpy() for x in self.model.trainable_variables]
        else:
            self.gradients.append([x.cpu().detach().numpy() for x in data[self.grad_key]])
            self.new_weight = [deepcopy(x).cpu().detach().numpy() for x in self.model.parameters() if x.requires_grad]

        if self.system.global_step % self.merge_grad == 0:
            self.test_self.assertTrue(self.check_if_changed())
            self.test_self.assertTrue(self.check_if_update_is_gradients())
        else:
            self.test_self.assertFalse(self.check_if_changed())  # don't change the model weight

        self.previous_weights = self.new_weight

    def check_if_changed(self):
        for var1, var2 in zip(self.previous_weights, self.new_weight):
            if not np.allclose(var1, var2):
                return True
        return False

    def check_if_update_is_gradients(self):
        for i in range(len(self.previous_weights)):
            diff = self.new_weight[i] - self.previous_weights[i]
            diff2 = sum([x[i] for x in self.gradients]) / self.merge_grad * -self.lr
            diff3 = sum([x[i] for x in self.gradients]) / self.merge_grad * -10.0
            if not np.allclose(diff, diff2, atol=1e-4):
                return False
        return True


class TestUpdateOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_data, _ = mnist.load_data()

    def test_tf_end_to_end(self):
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

            model = fe.build(model_fn=LeNet_tf,
                             optimizer_fn=lambda: tf.optimizers.SGD(lr),
                             mixed_precision=mixed_precision)
            network = fe.Network(ops=[
                ModelOp(model=model, inputs="x", outputs="y_pred"),
                CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                GradientOp(model=model, finals="ce", outputs="grad"),
                UpdateOp(model=model, loss_name="ce", gradients=gradient, merge_grad=merge_grad),
            ])

            traces = [
                CheckNetworkWeight(model=model,
                                   grad_key="grad",
                                   merge_grad=merge_grad,
                                   test_self=self,
                                   lr=lr,
                                   framework="tf")
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=2,
                                     traces=traces,
                                     max_train_steps_per_epoch=10)
            estimator.fit()

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if mixed_precision and gradient:
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_torch_end_to_end(self):
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])

            model = fe.build(model_fn=LeNet_torch,
                             optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=lr),
                             mixed_precision=mixed_precision)
            network = fe.Network(ops=[
                ModelOp(model=model, inputs="x", outputs="y_pred"),
                CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                GradientOp(model=model, finals="ce", outputs="grad"),
                UpdateOp(model=model, loss_name="ce", gradients=gradient, merge_grad=merge_grad),
            ])

            traces = [
                CheckNetworkWeight(model=model,
                                   grad_key="grad",
                                   merge_grad=merge_grad,
                                   test_self=self,
                                   lr=lr,
                                   framework="torch")
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=2,
                                     traces=traces,
                                     max_train_steps_per_epoch=10)
            estimator.fit()

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if mixed_precision and gradient:
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)
