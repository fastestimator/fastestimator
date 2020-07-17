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
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy as KerasCrossentropy

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.op.numpyop.univariate import Minmax
from fastestimator.op.tensorop.augmentation import MixUpBatch
from fastestimator.op.tensorop.loss import MixUpLoss, SparseCategoricalCrossentropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import Scheduler
from fastestimator.trace.io import ModelSaver
from fastestimator.trace.metric import Accuracy, ConfusionMatrix


def get_estimator(epochs=10, batch_size=32, alpha=1.0):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    num_classes = 10
    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=Minmax(inputs="x", outputs="x"))

    # step 2: prepare network
    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam")

    network = fe.Network(ops=[
        MixUpBatch(inputs="x", outputs=["x", "lambda"], alpha=alpha, mode="train"),
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        MixUpLoss(KerasCrossentropy(), lam="lambda", y_true="y", y_pred="y_pred", mode="train", outputs="loss"),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", mode="eval", outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred")
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
