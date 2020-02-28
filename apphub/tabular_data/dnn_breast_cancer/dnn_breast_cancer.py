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
import fastestimator as fe
import tensorflow as tf
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.trace.io import ModelSaver
from fastestimator.pipeline import Pipeline
from fastestimator.estimator import Estimator
from fastestimator.op.tensorop.loss import MeanSquaredError
from fastestimator.trace.metric import Accuracy


def create_dnn():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(30, )))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    return model


def get_estimator(epochs=50, batch_size=32, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    data = {"train": train_data, "eval": eval_data}
    
    pipeline = Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        test_data=test_data,
                        batch_size=batch_size)

    # step 2. prepare model
    model = fe.build(model=create_dnn, optimizer="adam")
    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"), MeanSquaredError(inputs=("y", "y_pred"), outputs="loss")
    ])

    # step 3.prepare estimator
    estimator = Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             log_steps=10,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()