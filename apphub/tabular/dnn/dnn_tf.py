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

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import fastestimator as fe
from fastestimator.dataset import breast_cancer
from fastestimator.estimator import Estimator
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def create_dnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(30, )))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


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
    model = fe.build(model_fn=create_dnn, optimizer_fn="adam")
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
