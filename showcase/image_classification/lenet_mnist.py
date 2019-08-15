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

from fastestimator.architecture import LeNet
from fastestimator.estimator.trace import Accuracy
from fastestimator.network.loss import SparseCategoricalCrossentropy
from fastestimator.network.model import ModelOp, build
from fastestimator.pipeline.processing import Minmax


def get_estimator():
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    data = {
        "train": {
            "x": np.expand_dims(x_train, -1), "y": y_train
        },
        "eval": {
            "x": np.expand_dims(x_eval, -1), "y": y_eval
        }
    }
    pipeline = fe.Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))
    # step 2. prepare model
    model = build(keras_model=LeNet(),
                  loss=SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred"),
                  optimizer="adam")
    network = fe.Network(ops=ModelOp(inputs="x", model=model, outputs="y_pred"))
    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator
