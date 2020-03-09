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
"""
The FastEstimator implementation of cifar10-fast model
ref: https://github.com/davidcpage/cifar10-fast
"""
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop import ChannelTranspose, CoarseDropout, HorizontalFlip, Normalize, Onehot, PadIfNeeded, \
    RandomCrop, Sometimes
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class FastCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.8)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.8)
        self.residual1 = Residual(128, 128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.8)
        self.residual2 = Residual(256, 256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.8)
        self.residual3 = Residual(512, 512)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = fn.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = fn.softmax(x, dim=-1)
        return x


class Residual(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x


def get_estimator(epochs=24, batch_size=512, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x"),
            Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.2)
        ])

    # step 2: prepare network
    model = fe.build(model_fn=FastCifar, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3 prepare estimator
    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=[
            Accuracy(true_key="y", pred_key="y_pred"),
            BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
        ],
        max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
