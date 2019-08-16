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
import numpy as np
import tensorflow as tf
import fastestimator as fe

from tensorflow.python.keras import layers
from fastestimator.pipeline.processing import Reshape
from fastestimator.network.model import ModelOp, build
from fastestimator.network.loss import BinaryCrossentropy
from fastestimator.estimator.trace import Accuracy


MAX_WORDS = 1000
MAX_LEN = 300


def create_lstm():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(MAX_WORDS, 100, input_length=MAX_LEN))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def pad(input_list, padding_size, padding_value):
    return input_list + [padding_value] * abs((len(input_list) - padding_size))


def get_estimator(epochs=10, batch_size=64, optimizer="adam"):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=MAX_LEN, num_words=MAX_WORDS)
    data = {
        "train": {
            "x": np.array([pad(x, MAX_LEN, 0) for x in x_train]),
            "y": y_train
        },
        "eval": {
            "x": np.array([pad(x, MAX_LEN, 0) for x in x_eval]),
            "y": y_eval
        }
    }

    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Reshape([1], inputs="y", outputs="y"))

    # step 2. prepare model
    model = build(keras_model=create_lstm(),
                  loss=BinaryCrossentropy(y_true="y", y_pred="y_pred"),
                  optimizer=optimizer)
    network = fe.Network(ops=ModelOp(inputs="x", model=model, outputs="y_pred"))

    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))

    return estimator


if __name__ == "__main__":
    o = get_estimator()
    o.fit()
