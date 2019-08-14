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
import tensorflow as tf

from fastestimator import Estimator
from fastestimator import Network
from fastestimator import Pipeline
from fastestimator.architecture import LeNet
from fastestimator.estimator.trace import Accuracy, ConfusionMatrix
from fastestimator.network.loss import SparseCategoricalCrossentropy
from fastestimator.network.model import build, AdverseModelOp
from fastestimator.pipeline.processing import Minmax


def get_estimator(epochs=2, batch_size=32, epsilon=0.1, warmup=1000):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    num_classes = 10

    pipeline = Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))

    model = build(keras_model=LeNet(input_shape=x_train.shape[1:], classes=num_classes),
                  loss=SparseCategoricalCrossentropy(true_key="y", pred_key="y_pred"), optimizer="adam")

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        ConfusionMatrix(true_key="y", pred_key="y_pred", num_classes=num_classes)
    ]

    network = Network(ops=AdverseModelOp(inputs=("x",
                                                 "y"), model=model, outputs="y_pred", epsilon=epsilon, warmup=warmup))

    estimator = Estimator(network=network, pipeline=pipeline, epochs=epochs, traces=traces)

    return estimator
