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
from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class Network:
    def __init__(self):
        self.model = self.create_dnn()
        self.optimizer = tf.optimizers.Adam(learning_rate=0.1)
        self.loss = tf.losses.MeanSquaredError()

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

    def create_dnn(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation="linear"))
        return model

def get_estimator(epochs=30, batch_size=32):

    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)

    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train= [[], []])

    estimator = Estimator(network= Network(),
                          pipeline=pipeline,
                          epochs= epochs,
                          log_steps=10)
    return estimator
