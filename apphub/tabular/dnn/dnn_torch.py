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
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.preprocessing import StandardScaler

import fastestimator as fe
from fastestimator.dataset import breast_cancer
from fastestimator.estimator import Estimator
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class DNN(torch.nn.Module):
    def __init__(self, num_inputs=30, n_outputs=1):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 16)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(16, 8)
        self.dp3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(8, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = fn.relu(x)
        x = self.dp2(x)
        x = self.fc3(x)
        x = fn.relu(x)
        x = self.dp3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


def get_estimator(batch_size=32, save_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    train_data, eval_data = breast_cancer.load_data()
    test_data = eval_data.split(0.5)

    # Apply some global pre-processing to the data
    scaler = StandardScaler()
    train_data["x"] = scaler.fit_transform(train_data["x"])
    eval_data["x"] = scaler.transform(eval_data["x"])
    test_data["x"] = scaler.transform(test_data["x"])

    pipeline = Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data, batch_size=batch_size)

    # step 2. prepare model
    model = fe.build(model_fn=DNN, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3.prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = Estimator(pipeline=pipeline, network=network, epochs=20, log_steps=10, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
