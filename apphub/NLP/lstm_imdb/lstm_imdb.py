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
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.op.tensorop import BinaryCrossentropy, ModelOp, Reshape
from fastestimator.trace import Accuracy, ModelSaver

MAX_WORDS = 10000
MAX_LEN = 500


def create_lstm(max_len):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(MAX_WORDS, 64, input_length=max_len))
    model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def pad(input_list, padding_size, padding_value):
    return input_list + [padding_value] * abs((len(input_list) - padding_size))


def get_estimator(epochs=10, batch_size=64, max_len=500, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=MAX_WORDS)
    data = {
        "train": {
            "x": np.array([pad(x, max_len, 0) for x in x_train]), "y": y_train
        },
        "eval": {
            "x": np.array([pad(x, max_len, 0) for x in x_eval]), "y": y_eval
        }
    }

    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Reshape([1], inputs="y", outputs="y"))

    # step 2. prepare model
    model = fe.build(model_def=lambda: create_lstm(max_len), model_name="lstm_imdb", optimizer="adam", loss_name="loss")
    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        BinaryCrossentropy(y_true="y", y_pred="y_pred", outputs="loss")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        ModelSaver(model_name="lstm_imdb", save_dir=model_dir, save_best=True)
    ]
    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
