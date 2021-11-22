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
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.backend import binary_crossentropy
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop import LambdaOp
from fastestimator.op.numpyop.univariate import ExpandDims, Normalize
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver


class GLoss(TensorOp):
    def forward(self, data, state):
        return binary_crossentropy(y_pred=data, y_true=torch.ones_like(data), from_logits=True)


class DLoss(TensorOp):
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = binary_crossentropy(y_pred=true_score, y_true=torch.ones_like(true_score), from_logits=True)
        fake_loss = binary_crossentropy(y_pred=fake_score, y_true=torch.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 7 * 7 * 256, bias=False)
        self.bn1d = nn.BatchNorm1d(7 * 7 * 256)
        self.conv_tran1 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(128)
        self.conv_tran2 = nn.ConvTranspose2d(128, 64, 5, stride=2, bias=False, padding=2, output_padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv_tran3 = nn.ConvTranspose2d(64, 1, 5, stride=2, bias=False, padding=2, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1d(x)
        x = fn.leaky_relu(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_tran1(x)
        x = self.bn2d1(x)
        x = fn.leaky_relu(x)
        x = self.conv_tran2(x)
        x = self.bn2d2(x)
        x = fn.leaky_relu(x)
        x = self.conv_tran3(x)
        x = torch.tanh(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 128, 1)
        self.dp1 = nn.Dropout2d(0.3)
        self.dp2 = nn.Dropout2d(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.leaky_relu(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = fn.leaky_relu(x)
        x = self.dp2(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        return x


def get_estimator(epochs=50, batch_size=256, max_train_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    train_data, _ = mnist.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x", axis=0),
            Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5),
            LambdaOp(fn=lambda: np.random.normal(size=[100]).astype('float32'), outputs="z")
        ])
    gen_model = fe.build(model_fn=Generator, optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-4))
    disc_model = fe.build(model_fn=Discriminator, optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-4))
    network = fe.Network(ops=[
        ModelOp(model=gen_model, inputs="z", outputs="x_fake"),
        ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
        GLoss(inputs="fake_score", outputs="gloss"),
        UpdateOp(model=gen_model, loss_name="gloss"),
        ModelOp(inputs="x", model=disc_model, outputs="true_score"),
        DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
        UpdateOp(model=disc_model, loss_name="dloss")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=ModelSaver(model=gen_model, save_dir=save_dir, frequency=5),
                             train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
