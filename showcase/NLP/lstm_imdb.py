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
from fastestimator.pipeline.static.preprocess import Reshape
from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.estimator.trace import Accuracy
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

MAX_WORDS = 1000
MAX_LEN = 150

class Network:
    def __init__(self):
        self.model = self.create_lstm()
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.BinaryCrossentropy()

    def train_op(self, batch):
        with tf.GradientTape() as tape:
            predictions = self.model(batch["x"])
            loss = self.loss(batch["y"], predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions, loss

    def eval_op(self, batch):
        predictions = self.model(batch["x"], training=False)
        loss = self.loss(batch["y"], predictions)
        return predictions, loss

    def create_lstm(self):
        model = tf.keras.Sequential()
        model.add(layers.Embedding(MAX_WORDS, 100, input_length=MAX_LEN))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv1D(64, 5, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=4))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

def pad(list, padding_size, padding_value):
    return list + [padding_value] * abs((len(list)-padding_size))

def get_estimator(epochs=10, batch_size=64, optimizer="adam"):

    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=300, num_words=MAX_WORDS)
    x_train = np.array([pad(x, 300, 0) for x in x_train])
    x_eval = np.array([pad(x, 300, 0) for x in x_eval])

    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train= [[], [Reshape([1])]])

    traces = [Accuracy(y_true_key="y")]

    estimator = Estimator(network= Network(),
                          pipeline=pipeline,
                          epochs= epochs,
                          traces= traces)
    return estimator