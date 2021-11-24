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
from fastestimator.schedule import EpochScheduler, RepeatScheduler


class CheckNetworkWeight(fe.trace.Trace):
    """This class will test if the model weight updates correctly.

    Args:
        model: Target model.
        grad_key: Gradients key.
        merge_grad: 'merge_grad' argument of UpdateOp.
        test_self: The object reference of unittest.TestClass
        framework: Which framework the testing target uses. ["tf", "torch"]
        lrs: Learning rate of the optimizer. LR is constant if `lrs` is not iterable. Otherwise the lrs will be
            lrs[x] when work_intervals[x][0] <= epoch < epoch work_intervals[x][1]
        work_intervals: At which interval of epoch does the test activated. This argument need to work with `lrs`.
            When epoch is not in the work_intervals, the test will be skipped. If None, testing will be active all the
            time.
    """
    def __init__(self, model, grad_key, merge_grad, test_self, framework, lrs, work_intervals=None):
        if work_intervals:
            assert len(work_intervals) == len(lrs), "length of work_intervals need to be the same as lrs"
        else:
            assert isinstance(lrs, (int, float)), "if work_intervals is None,  lrs need to be a value"

        super().__init__(inputs=grad_key)
        self.model = model
        self.grad_key = grad_key
        self.merge_grad = merge_grad
        self.test_self = test_self
        self.framework = framework
        self.lrs = lrs
        self.work_intervals = work_intervals
        self.n_gpu = torch.cuda.device_count()
        if self.framework == "tf":
            self.previous_weights = [x.numpy() for x in model.trainable_variables]
        else:
            self.previous_weights = [
                deepcopy(x).cpu().detach().numpy() for x in self.model.parameters() if x.requires_grad
            ]
        self.gradients = deque(maxlen=merge_grad)
        self.new_weight = None

    def on_batch_end(self, data):
        if self.framework == "tf":
            if self.n_gpu > 1:
                # the data[self.key] shape is self.n_gpu times large on axis 0 and need to be folded
                gradients = [self.fold(x.numpy()) for x in data[self.grad_key]]
            else:
                gradients = [x.numpy() for x in data[self.grad_key]]

            self.gradients.append(gradients)
            self.new_weight = [x.numpy() for x in self.model.trainable_variables]
        else:
            self.gradients.append([x.cpu().detach().numpy() for x in data[self.grad_key]])
            self.new_weight = [deepcopy(x).cpu().detach().numpy() for x in self.model.parameters() if x.requires_grad]

        lr = self.get_lr()  # if lr is False, don't need to do the check
        if lr:
            if self.system.global_step % self.merge_grad == 0:
                self.test_self.assertTrue(self.check_if_changed())  # model weight should change
                self.test_self.assertTrue(self.check_if_update_is_gradients(lr))  # model weight should be correct
            else:
                self.test_self.assertFalse(self.check_if_changed())  # model weight should not change

        self.previous_weights = self.new_weight

    def check_if_changed(self):
        """Check if model weight get updated
        """
        for var1, var2 in zip(self.previous_weights, self.new_weight):
            if not np.allclose(var1, var2):
                return True
        return False

    def check_if_update_is_gradients(self, lr):
        """Check if the updated model weight is correct
        """
        for i in range(len(self.previous_weights)):
            diff = self.new_weight[i] - self.previous_weights[i]
            diff2 = sum([x[i] for x in self.gradients]) / self.merge_grad * -lr

            if not np.allclose(diff, diff2, atol=1e-4):
                return False
        return True

    def get_lr(self):
        """Get the current lr. Return False if testing is not active.
        """
        if not self.work_intervals:
            return self.lrs

        for lr, interval in zip(self.lrs, self.work_intervals):
            assert len(interval) == 2
            if interval[0] <= self.system.epoch_idx < interval[1]:
                return lr

        return False

    def fold(self, x):
        """Reduce the size of `x` on axis 0 by self.n_gpu factor.
        """
        assert isinstance(x, np.ndarray)
        assert x.shape[0] % self.n_gpu == 0
        len_org = x.shape[0]
        len_new = int(x.shape[0] / self.n_gpu)
        return sum([x[start:start + len_new] / self.n_gpu for start in range(0, len_org, len_new)])


class TestUpdateOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_data, _ = mnist.load_data()

    def test_tf_end_to_end(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
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
                                   lrs=lr,
                                   framework="tf")
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=2,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_torch_end_to_end(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
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
                                   lrs=lr,
                                   framework="torch")
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=2,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)
                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_tf_multi_optimizer_with_epoch_scheduler(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            lr2 = 0.01
            lr3 = 0.001
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

            optimizer_fn = EpochScheduler({
                1: lambda: tf.optimizers.SGD(lr), 2: lambda: tf.optimizers.SGD(lr2), 3: lambda: tf.optimizers.SGD(lr3)
            })

            model = fe.build(model_fn=LeNet_tf, optimizer_fn=optimizer_fn, mixed_precision=mixed_precision)
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
                                   framework="tf",
                                   lrs=[lr, lr2, lr3],
                                   work_intervals=[[1, 2], [2, 3], [3, 4]])
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=3,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_torch_multi_optimizer_with_epoch_scheduler(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            lr2 = 0.01
            lr3 = 0.001
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])

            optimizer_fn = EpochScheduler({
                1: lambda x: torch.optim.SGD(params=x, lr=lr),
                2: lambda x: torch.optim.SGD(params=x, lr=lr2),
                3: lambda x: torch.optim.SGD(params=x, lr=lr3)
            })

            model = fe.build(model_fn=LeNet_torch, optimizer_fn=optimizer_fn, mixed_precision=mixed_precision)
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
                                   framework="torch",
                                   lrs=[lr, lr2, lr3],
                                   work_intervals=[[1, 2], [2, 3], [3, 4]])
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=3,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_tf_multi_optimizer_with_repeat_scheduler(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            lr2 = 0.01
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

            optimizer_fn = RepeatScheduler([lambda: tf.optimizers.SGD(lr), lambda: tf.optimizers.SGD(lr2)])

            model = fe.build(model_fn=LeNet_tf, optimizer_fn=optimizer_fn, mixed_precision=mixed_precision)
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
                                   framework="tf",
                                   lrs=[lr, lr2, lr, lr2],
                                   work_intervals=[[1, 2], [2, 3], [3, 4], [4, 5]])
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=4,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)

    def test_torch_multi_optimizer_with_repeat_scheduler(self):
        """This test cover the all combination of:
            - mixed-precision / not
            - merge_grad / not
            - gradient input / loss input
        """
        def run_test(mixed_precision, merge_grad, gradient):
            lr = 0.1
            lr2 = 0.01
            pipeline = fe.Pipeline(train_data=self.train_data,
                                   batch_size=4,
                                   ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])

            optimizer_fn = RepeatScheduler(
                [lambda x: torch.optim.SGD(params=x, lr=lr), lambda x: torch.optim.SGD(params=x, lr=lr2)])

            model = fe.build(model_fn=LeNet_torch, optimizer_fn=optimizer_fn, mixed_precision=mixed_precision)
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
                                   framework="torch",
                                   lrs=[lr, lr2, lr, lr2],
                                   work_intervals=[[1, 2], [2, 3], [3, 4], [4, 5]])
            ]
            estimator = fe.Estimator(pipeline=pipeline,
                                     network=network,
                                     epochs=4,
                                     traces=traces,
                                     train_steps_per_epoch=2)
            estimator.fit(warmup=False)

        for mixed_precision in [True, False]:
            for merge_grad in [1, 2]:
                for gradient in ["grad", None]:
                    with self.subTest("mixed_precision: {}, merge_grad: {}, take: {}".format(
                            mixed_precision, merge_grad, "gradient" if gradient else "loss")):

                        if (mixed_precision and gradient) or (torch.cuda.device_count() > 1 and merge_grad > 1):
                            with self.assertRaises(ValueError):
                                run_test(mixed_precision, merge_grad, gradient)

                        else:
                            run_test(mixed_precision, merge_grad, gradient)
