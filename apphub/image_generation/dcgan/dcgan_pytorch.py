# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
"""DCGAN example using MNIST data set."""
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as fn
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.backend import cross_entropy
from fastestimator.dataset import mnist
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import ExpandDims, Normalize
from fastestimator.op.tensorop.model import ModelOp, UpdateOp


class LeNet(torch.nn.Module):
    def __init__(self, n_channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = fn.softmax(x, dim=-1)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7 * 7 * 256, bias=False)
        self.bn1d = nn.BatchNorm1d(7 * 7 * 256)
        self.conv_tran1 = nn.ConvTranspose2d(256, 128, 5, bias=False)
        self.bn2d1 = nn.BatchNorm2d(128)
        self.conv_tran2 = nn.ConvTranspose2d(128, 64, 5, bias=False)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.convtrans3 = nn.ConvTranspose2d(64, 1, 5, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = fn.leaky_relu(x)
        


def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


class GLoss(TensorOp):
    def forward(self, data, state):
        return cross_entropy(y_pred=data, y_true=tf.ones_like(data), from_logits=True)


class DLoss(TensorOp):
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = cross_entropy(y_pred=true_score, y_true=tf.ones_like(true_score), from_logits=True)
        fake_loss = cross_entropy(y_pred=fake_score, y_true=tf.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss


def discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


def get_estimator(batch_size=256, epochs=50):
    train_data, _ = mnist.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x", axis=0),
            Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5)
        ])
    gen_model = fe.build(model_fn=generator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    disc_model = fe.build(model_fn=discriminator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(model=gen_model, inputs=lambda: tf.random.normal([batch_size, 100]), outputs="x_fake"),
        ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
        ModelOp(inputs="x", model=disc_model, outputs="true_score"),
        GLoss(inputs="fake_score", outputs="gloss"),
        UpdateOp(model=gen_model, loss_name="gloss"),
        DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
        UpdateOp(model=disc_model, loss_name="dloss")
    ])
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
