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

import torch.nn as nn
import torch.nn.functional as fn

import fastestimator as fe
from fastestimator.dataset.data import imdb_review
from fastestimator.op.numpyop.univariate import Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class ReviewSentiment(nn.Module):
    def __init__(self, max_words, embedding_size=64, hidden_units=64):
        super().__init__()
        self.embedding = nn.Embedding(max_words, embedding_size)
        self.conv1d = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=4)
        self.lstm = nn.LSTM(input_size=125, hidden_size=hidden_units, num_layers=1)
        self.fc1 = nn.Linear(in_features=hidden_units, out_features=250)
        self.fc2 = nn.Linear(in_features=250, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute((0, 2, 1))
        x = self.conv1d(x)
        x = fn.relu(x)
        x = self.maxpool1d(x)
        output, _ = self.lstm(x)
        x = output[:, -1]  # sequence output of only last timestamp
        x = fn.tanh(x)
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = fn.sigmoid(x)
        return x


def get_estimator(max_words=10000,
                  max_len=500,
                  epochs=10,
                  batch_size=64,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    train_data, eval_data = imdb_review.load_data(max_len, max_words)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=Reshape(1, inputs="y", outputs="y"))
    # step 2. prepare model
    model = fe.build(model_fn=lambda: ReviewSentiment(max_words=max_words), optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])
    # step 3.prepare estimator
    traces = [Accuracy(true_key="y", pred_key="y_pred"), BestModelSaver(model=model, save_dir=save_dir)]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             eval_steps_per_epoch=max_eval_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
