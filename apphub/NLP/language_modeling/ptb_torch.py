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

import numpy as np
import torch
import torch.nn as nn

import fastestimator as fe
from fastestimator.dataset.data.penn_treebank import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import EarlyStopping, LRScheduler
from fastestimator.trace.io import BestModelSaver


class CreateInputAndTarget(NumpyOp):
    def forward(self, data, state):
        return data[:-1], data[1:]


class DimesionAdjust(TensorOp):
    def forward(self, data, state):
        x, y = data
        return x.T, y.T.reshape(-1)


class Perplexity(fe.trace.Trace):
    def on_epoch_end(self, data):
        ce = data["ce"]
        data.write_with_log(self.outputs[0], np.exp(ce))


class BuildModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=300, rnn_units=600):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, rnn_units)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(rnn_units, vocab_size)

        nn.init.xavier_uniform_(self.lstm_layer.weight_ih_l0.data)
        nn.init.xavier_uniform_(self.lstm_layer.weight_hh_l0.data)

    def forward(self, x):
        x = self.embed_layer(x)
        x, _ = self.lstm_layer(x)
        x = x.view(x.size(0) * x.size(1), x.size(2))
        x = self.dropout(x)
        x = self.fc(x)
        return x


def lr_schedule(step, init_lr):
    if step <= 1725:
        lr = init_lr + (0.7 * init_lr * (step - 1) / 1725)
    else:
        lr = max(1.7 * init_lr * ((2415 - step + 1725) / 2415), 0.01)
    return lr


def get_estimator(epochs=30,
                  batch_size=128,
                  seq_length=20,
                  vocab_size=10000,
                  data_dir=None,
                  max_train_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    train_data, eval_data, _, _ = load_data(root_dir=data_dir, seq_length=seq_length + 1)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=CreateInputAndTarget(inputs="x", outputs=("x", "y")),
                           drop_last=True)
    # step 2
    model = fe.build(model_fn=lambda: BuildModel(vocab_size, embedding_dim=300, rnn_units=600),
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=1.0, momentum=0.9))
    network = fe.Network(ops=[
        DimesionAdjust(inputs=("x", "y"), outputs=("x", "y")),
        ModelOp(model=model, inputs="x", outputs="y_pred", mode=None),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", form="sparse", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Perplexity(inputs="ce", outputs="perplexity", mode="eval"),
        LRScheduler(model=model, lr_fn=lambda step: lr_schedule(step, init_lr=1.0)),
        BestModelSaver(model=model, save_dir=save_dir, metric='perplexity', save_best_mode='min', load_best_final=True),
        EarlyStopping(monitor="perplexity", patience=5)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
