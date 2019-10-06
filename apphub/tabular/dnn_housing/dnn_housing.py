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
from sklearn.preprocessing import StandardScaler

import fastestimator as fe
import tensorflow as tf
from fastestimator.op.tensorop import MeanSquaredError, ModelOp
from fastestimator.trace import ModelSaver
from tensorflow.keras import layers


def create_dnn():
    model = tf.keras.Sequential()

    model.add(layers.Dense(23, activation="relu", input_shape=(13, )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="linear"))

    return model


def get_estimator(epochs=50, batch_size=32, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.boston_housing.load_data()

    # step 1. prepare data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_eval = scaler.transform(x_eval)
    train_data = {"x": x_train, "y": np.expand_dims(y_train, -1)}
    eval_data = {"x": x_eval, "y": np.expand_dims(y_eval, -1)}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data)

    # step 2. prepare model
    model = fe.build(model_def=create_dnn, model_name="dnn", optimizer="adam", loss_name="loss")
    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"), MeanSquaredError(inputs=("y", "y_pred"), outputs="loss")
    ])

    # step 3.prepare estimator
    traces = [ModelSaver(model_name="dnn", save_dir=model_dir, save_best=True)]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps,
                             log_steps=10,
                             traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
