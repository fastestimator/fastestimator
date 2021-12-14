# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as fn
from tensorflow.keras import Sequential, initializers, layers

import fastestimator as fe
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy, L2Regularizaton
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


class MyNet_torch(torch.nn.Module):
    """A standard DNN implementation in pytorch.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    The MyNet model has 3 dense layers.

    Args:
        input_shape: The shape of the model input (channels, height, width).
        classes: The number of outputs the model should generate.
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 28, 28), classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_shape[1] * input_shape[2], 300)
        self.fc1.weight.data.fill_(0.01)
        self.fc1.bias.data.fill_(0.0)

        self.fc2 = nn.Linear(300, 64)
        self.fc2.weight.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.0)

        self.fc3 = nn.Linear(64, classes)
        self.fc3.weight.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = fn.softmax(self.fc3(x), dim=-1)
        return x


def MyNet_tf(input_shape: Tuple[int, int, int] = (28, 28, 1), classes: int = 10) -> tf.keras.Model:
    """A standard DNN implementation in TensorFlow.

    The MyNet model has 3 dense layers.

    Args:
        input_shape: shape of the input data (height, width, channels).
        classes: The number of outputs the model should generate.

    Returns:
        A TensorFlow MyNet model.
    """
    #_check_input_shape(input_shape)
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(
        layers.Dense(units=300,
                     kernel_initializer=initializers.Constant(0.01),
                     bias_initializer=initializers.Zeros(),
                     activation='relu'))
    model.add(
        layers.Dense(units=64,
                     kernel_initializer=initializers.Constant(0.01),
                     bias_initializer=initializers.Zeros(),
                     activation='relu'))
    model.add(
        layers.Dense(units=classes,
                     kernel_initializer=initializers.Constant(0.01),
                     bias_initializer=initializers.Zeros(),
                     activation='softmax'))

    return model


class TestL2Regularization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Weight Decay pytorch model
        cls.beta = 0.1

    def test_l2_regularization_pytorch(self):
        pytorch_l2 = fe.build(model_fn=MyNet_torch, optimizer_fn="adam")
        l2 = L2Regularizaton(inputs='x', outputs='x', model=pytorch_l2)
        output = l2.forward(data=torch.tensor([0.0]), state={})
        output = output.detach()
        self.assertTrue(np.allclose(output.numpy(), 0.12751994))

    def test_l2_regularization_tensorflow(self):
        tf_l2 = fe.build(model_fn=MyNet_tf, optimizer_fn="adam")
        l2 = L2Regularizaton(inputs='x', outputs='x', model=tf_l2)
        output = l2.forward(data=tf.constant([0.0]), state={})
        self.assertTrue(np.allclose(output.numpy(), 0.1275205))

    def test_pytorch_weight_decay_vs_l2(self):
        # Get Data
        train_data, _ = mnist.load_data()
        t_d = train_data.split(128)
        # Initializing models
        pytorch_wd = fe.build(model_fn=MyNet_torch,
                              optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01, weight_decay=self.beta))

        pytorch_l2 = fe.build(model_fn=MyNet_torch, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01))
        # Initialize pipeline
        pipeline = fe.Pipeline(train_data=t_d,
                               batch_size=128,
                               ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
        # Define the two pytorch networks
        network_weight_decay = fe.Network(ops=[
            ModelOp(model=pytorch_wd, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=pytorch_wd, loss_name="ce")
        ])

        network_l2 = fe.Network(ops=[
            ModelOp(model=pytorch_l2, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            L2Regularizaton(inputs="ce", outputs="l2", model=pytorch_l2, beta=self.beta),
            UpdateOp(model=pytorch_l2, loss_name="l2")
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
        for wt, l2 in zip(pytorch_wd.parameters(), pytorch_l2.parameters()):
            if ((wt - l2).abs()).sum() < torch.tensor(10**-6):
                count += 1
        self.assertTrue(count == 6)

    def test_pytorch_l2_vs_tensorflow_l2(self):
        # Get Data
        train_data, eval_data = mnist.load_data()
        t_d = train_data.split(128)
        # Initializing Pytorch model
        pytorch_l2 = fe.build(model_fn=MyNet_torch, optimizer_fn=lambda x: torch.optim.SGD(params=x, lr=0.01))
        # Initialize Pytorch pipeline
        pipeline = fe.Pipeline(train_data=t_d,
                        eval_data=eval_data,
                        batch_size=128,
                        ops=[ExpandDims(inputs="x", outputs="x", axis=0),
                                Minmax(inputs="x", outputs="x")])
        # Initialize Pytorch Network
        network_l2 = fe.Network(ops=[
            ModelOp(model=pytorch_l2, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            L2Regularizaton(inputs="ce",outputs="l2",model=pytorch_l2,beta = self.beta),
            UpdateOp(model=pytorch_l2, loss_name="l2")
        ])
        # step 3
        traces = [
            Accuracy(true_key="y", pred_key="y_pred")
        ]
        # Initialize Pytorch estimator
        estimator_l2 = fe.Estimator(pipeline=pipeline,
                            network=network_l2,
                            epochs=1,
                            traces=traces,
                            train_steps_per_epoch=1,
                            monitor_names=["ce","l2"])
        print('********************************Pytorch L2 Regularization training************************************')
        estimator_l2.fit()

        # Converting Pytorch weights to numpy
        torch_wt = []
        for _, param in pytorch_l2.named_parameters():
            if param.requires_grad:
                torch_wt.append(param.detach().numpy())

        # step 1
        pipeline = fe.Pipeline(train_data=t_d,
                               eval_data=eval_data,
                               batch_size=128,
                               ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
        # step 2
        model_tf = fe.build(model_fn=MyNet_tf, optimizer_fn=lambda: tf.optimizers.SGD(learning_rate=0.01))
        network = fe.Network(ops=[
            ModelOp(model=model_tf, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            L2Regularizaton(inputs="ce",outputs="l2",model=model_tf,beta = self.beta),
            UpdateOp(model=model_tf, loss_name="l2")
        ])
        # step 3
        traces = [
            Accuracy(true_key="y", pred_key="y_pred")
        ]
        estimator = fe.Estimator(pipeline=pipeline,
                                network=network,
                                epochs=1,
                                traces=traces,
                                train_steps_per_epoch=1,
                                monitor_names=["ce","l2"])
        print('*******************************Tensorflow L2 Regularization training***********************************')
        estimator.fit()


        # Converting TF weights to numpy
        tf_wt = []
        for layer in model_tf.layers:
            for w in layer.trainable_variables:
                tf_wt.append(w.numpy())

        # testing weights
        count = 0
        for tf_t,tr in zip(tf_wt,torch_wt):
            if np.sum(np.abs(tf_t-np.transpose(tr))) < (10**-5):
                count += 1
        self.assertTrue(count == 6)
