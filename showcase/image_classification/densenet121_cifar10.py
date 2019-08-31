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
from tensorflow.keras.applications.densenet import DenseNet121 as DenseNet121_keras
from tensorflow.keras.layers import Dense, Input

import fastestimator as fe
from fastestimator.architecture import LeNet
from fastestimator.estimator.trace import Accuracy
from fastestimator.network.loss import SparseCategoricalCrossentropy
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.pipeline.processing import Minmax, Resize


def DenseNet121(input_shape, classes=10, weights=None):
    inputs = Input(input_shape)
    x = DenseNet121_keras(weights=weights, input_shape=input_shape, include_top=False, pooling='avg')(inputs)
    outputs = Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_estimator():
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    pipeline = fe.Pipeline(batch_size=64, data=data, ops=Minmax(inputs="x", outputs="x"))
    # step 2. prepare model
    model = FEModel(model_def=lambda: DenseNet121(input_shape=(32, 32, 3)), model_name="densenet121", optimizer="adam")

    network = fe.Network(ops=[
        ModelOp(inputs="x", fe_model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred")
    ])
    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=50,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator
