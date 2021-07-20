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
import tensorflow as tf

import fastestimator as fe
from fastestimator.dataset.data.penn_treebank import load_data
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.adapt import EarlyStopping, LRScheduler
from fastestimator.trace.io import BestModelSaver


class CreateInputAndTarget(NumpyOp):
    def forward(self, data, state):
        return data[:-1], data[1:]


class Perplexity(Trace):
    def on_epoch_end(self, data):
        ce = data["ce"]
        data.write_with_log(self.outputs[0], np.exp(ce))


def build_model(vocab_size, embedding_dim, rnn_units, seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[None, seq_length]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def lr_schedule(step, init_lr):
    if step <= 1725:
        lr = init_lr + init_lr * (step - 1) / 1725
    else:
        lr = max(2 * init_lr * ((6900 - step + 1725) / 6900), 1.0)
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
    model = fe.build(model_fn=lambda: build_model(vocab_size, embedding_dim=300, rnn_units=600, seq_length=seq_length),
                     optimizer_fn=lambda: tf.optimizers.SGD(1.0, momentum=0.9))

    network = fe.Network(ops=[
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
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
