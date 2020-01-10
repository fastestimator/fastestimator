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
"""This example showcase FastEstimator usage for pytorch users. In this file, we use data loader as data input.
"""
import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastestimator as fe
from fastestimator.pipeline import Pipeline, NumpyDataset
from fastestimator.op.numpyop import Minmax, ExpandDims
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

    def num_flat_features(self, x):
        return np.prod(x.size()[1:])


def get_estimator(batch_size=32):
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    train_data = NumpyDataset({"x": x_train, "y": y_train})
    eval_data = NumpyDataset({"x": x_eval, "y": y_eval})
    pipeline = Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=batch_size,
                        ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model=Net(), optimizer="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
