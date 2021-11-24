#  Copyright 2021 The FastEstimator Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import math
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data import cifair100
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, ChannelTranspose
from fastestimator.op.tensorop.loss import CrossEntropy, SuperLoss
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver, ImageSaver
from fastestimator.trace.metric import MCC
from fastestimator.trace.xai import LabelTracker


def swish(x):
    return x * torch.sigmoid(x)


class big_lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(32, momentum=0.01)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(64, momentum=0.01)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(128, momentum=0.01)
        self.fc1 = nn.Linear(2048, 128)
        self.fc1_bn = nn.BatchNorm1d(128, momentum=0.01)
        self.fc2 = nn.Linear(128, 100)

    def forward(self, x):
        # layer 0
        x = swish(self.conv0(x))
        x = self.conv0_bn(x)
        x = fn.max_pool2d(x, 2)
        # layer 1
        x = swish(self.conv1(x))
        x = self.conv1_bn(x)
        x = fn.max_pool2d(x, 2)
        # layer 2
        x = swish(self.conv2(x))
        x = self.conv2_bn(x)
        x = fn.max_pool2d(x, 2)
        # layer 3
        x = torch.flatten(x, 1)
        x = swish(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc2(x)
        x = fn.softmax(x, dim=-1)
        return x


def corrupt_dataset(dataset, n_classes=100, corruption_fraction=0.4):
    # Keep track of which samples were corrupted for visualization later
    corrupted = [0 for _ in range(len(dataset))]
    # Perform the actual label corruption
    n_samples_per_class = len(dataset) // n_classes
    n_to_corrupt_per_class = math.floor(corruption_fraction * n_samples_per_class)
    n_corrupted = [0] * n_classes
    i = 0
    while any([elem < n_to_corrupt_per_class for elem in n_corrupted]):
        current_class = dataset[i]['y'].item()
        if n_corrupted[current_class] < n_to_corrupt_per_class:
            dataset[i]['y'] = (dataset[i]['y'] + np.random.randint(1, n_classes)) % n_classes
            n_corrupted[current_class] += 1
            corrupted[i] = 1
        i += 1
    # Put the corruption labels into the dataset for visualization
    dataset['data_labels'] = np.array(corrupted, dtype=np.int).reshape((len(dataset), 1))


def get_estimator(epochs=50,
                  batch_size=128,
                  train_steps_per_epoch=None,
                  eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = cifair100.load_data()

    # Add label noise to simulate real-world labeling problems
    corrupt_dataset(train_data)

    test_data = eval_data.split(range(len(eval_data) // 2))
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])

    # step 2
    model = fe.build(model_fn=big_lenet, optimizer_fn='adam')
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SuperLoss(CrossEntropy(inputs=("y_pred", "y"), outputs="ce"), output_confidence="confidence"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3
    traces = [
        MCC(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="mcc", save_best_mode="max", load_best_final=True),
        LabelTracker(metric="confidence",
                     label="data_labels",
                     label_mapping={
                         "Normal": 0, "Corrupted": 1
                     },
                     mode="train",
                     outputs="label_confidence"),
        ImageSaver(inputs="label_confidence", save_dir=save_dir, mode="train"),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             eval_steps_per_epoch=eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
