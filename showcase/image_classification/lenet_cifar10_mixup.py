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
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy as KerasCrossentropy

from fastestimator import Estimator
from fastestimator import Network
from fastestimator import Pipeline
from fastestimator.architecture import LeNet
from fastestimator.estimator.trace import Accuracy, ConfusionMatrix
from fastestimator.network.loss import MixUpLoss, SparseCategoricalCrossentropy, Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.pipeline.augmentation import MixUpBatch
from fastestimator.pipeline.processing import Minmax


def get_estimator(epochs=2, batch_size=32, alpha=1.0, warmup=0):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    num_classes = 10
    pipeline = Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))

    model = build(keras_model=LeNet(input_shape=x_train.shape[1:], classes=num_classes),
                  loss=Loss(inputs="loss"),
                  optimizer="adam")

    network = Network(ops=[
        MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=alpha, warmup=warmup, mode="train"),
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        MixUpLoss(KerasCrossentropy(), lam="lambda", y_true="y", y_pred="y_pred", outputs="loss", mode="train"),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss", mode="eval")
    ])

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        ConfusionMatrix(true_key="y", pred_key="y_pred", num_classes=num_classes)
    ]

    estimator = Estimator(network=network, pipeline=pipeline, epochs=epochs, traces=traces)
    return estimator
