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
"""
The FastEstimator implementation of SimCLR with ResNet9 on CIFAR10.
This code took reference from google implementation (https://github.com/google-research/simclr).
Note that we use the ciFAIR10 dataset instead (https://cvjena.github.io/cifair/)
"""
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data.cifair10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, ColorJitter, GaussianBlur, ToFloat, ToGray
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver
from fastestimator.trace.metric import Accuracy


class ResNet9OneLayerHead(nn.Module):
    def __init__(self, length, input_size=(3, 32, 32)):
        super().__init__()
        self.encoder = ResNet9Encoder(input_size)
        self.fc1 = nn.Linear(512, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.fc1(x)
        return x


class ResNet9Encoder(nn.Module):
    def __init__(self, input_size=(3, 32, 32)):
        super().__init__()
        self.conv0 = nn.Conv2d(input_size[0], 64, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(64, momentum=0.2)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(128, momentum=0.2)
        self.residual1 = Residual(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(256, momentum=0.2)
        self.residual2 = Residual(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(512, momentum=0.2)
        self.residual3 = Residual(512)

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
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        return x


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x


class NTXentOp(TensorOp):
    def __init__(self, arg1, arg2, outputs, temperature=1.0, mode=None):
        super().__init__(inputs=(arg1, arg2), outputs=outputs, mode=mode)
        self.temperature = temperature

    def forward(self, data, state):
        arg1, arg2 = data
        loss, logit, label = NTXent(arg1, arg2, self.temperature)
        return loss, logit, label


def NTXent(A, B, temperature):
    device = A.device
    large_number = 1e9
    batch_size = A.shape[0]
    A = fn.normalize(A, p=2.0, dim=-1)
    B = fn.normalize(B, p=2.0, dim=-1)
    mask = torch.eye(batch_size).to(device)
    labels = torch.eye(batch_size, 2 * batch_size, dtype=torch.float).to(device)

    aa = torch.matmul(A, A.T) / temperature
    aa = aa - mask * large_number
    ab = torch.matmul(A, B.T) / temperature
    bb = torch.matmul(B, B.T) / temperature
    bb = bb - mask * large_number
    ba = torch.matmul(B, A.T) / temperature

    a_sim = torch.cat([ab, aa], dim=1)
    loss_a = torch.sum(-labels * torch.nn.LogSoftmax(dim=1)(a_sim), 1)

    b_sim = torch.cat([ba, bb], dim=1)
    loss_b = torch.sum(-labels * torch.nn.LogSoftmax(dim=1)(b_sim), 1)
    loss = torch.mean(loss_a + loss_b)

    return loss, ab, labels


def pretrain_model(epochs, batch_size, train_steps_per_epoch, save_dir):
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),

            # augmentation 1
            RandomCrop(32, 32, image_in="x", image_out="x_aug"),
            Sometimes(HorizontalFlip(image_in="x_aug", image_out="x_aug"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug", outputs="x_aug", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug", outputs="x_aug"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug", outputs="x_aug", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ChannelTranspose(inputs="x_aug", outputs="x_aug"),
            ToFloat(inputs="x_aug", outputs="x_aug"),

            # augmentation 2
            RandomCrop(32, 32, image_in="x", image_out="x_aug2"),
            Sometimes(HorizontalFlip(image_in="x_aug2", image_out="x_aug2"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug2", outputs="x_aug2", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug2", outputs="x_aug2"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug2", outputs="x_aug2", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ChannelTranspose(inputs="x_aug2", outputs="x_aug2"),
            ToFloat(inputs="x_aug2", outputs="x_aug2")
        ])

    model_con = fe.build(model_fn=lambda: ResNet9OneLayerHead(length=128), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model_con, inputs="x_aug", outputs="y_pred"),
        ModelOp(model=model_con, inputs="x_aug2", outputs="y_pred2"),
        NTXentOp(arg1="y_pred", arg2="y_pred2", outputs=["NTXent", "logit", "label"]),
        UpdateOp(model=model_con, loss_name="NTXent")
    ])

    traces = [
        Accuracy(true_key="label", pred_key="logit", mode="train", output_name="contrastive_accuracy"),
        ModelSaver(model=model_con, save_dir=save_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch)
    estimator.fit()

    return model_con


def finetune_model(model, epochs, batch_size, train_steps_per_epoch):
    train_data, test_data = load_data()
    train_data = train_data.split(0.1)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=test_data,
                           batch_size=batch_size,
                           ops=[
                               ToFloat(inputs="x", outputs="x"),
                               ChannelTranspose(inputs="x", outputs="x"),
                           ])

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=["y_pred", "y"], outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch)
    estimator.fit()


def fastestimator_run(epochs_pretrain=50,
                      epochs_finetune=10,
                      batch_size=512,
                      train_steps_per_epoch=None,
                      save_dir=tempfile.mkdtemp()):

    model_con = pretrain_model(epochs_pretrain, batch_size, train_steps_per_epoch, save_dir)
    model_finetune = fe.build(model_fn=lambda: ResNet9OneLayerHead(length=10), optimizer_fn="adam")
    model_finetune.encoder.load_state_dict(model_con.encoder.state_dict())  # load the encoder weight
    finetune_model(model_finetune, epochs_finetune, batch_size, train_steps_per_epoch)


if __name__ == "__main__":
    fastestimator_run()
