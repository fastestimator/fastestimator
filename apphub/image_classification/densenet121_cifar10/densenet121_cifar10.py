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
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Input

import fastestimator as fe
from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy
from fastestimator.trace import Accuracy, LRController, ModelSaver


def DenseNet121_cifar10():
    inputs = Input((32, 32, 3))
    x = DenseNet121(weights=None, input_shape=(32, 32, 3), include_top=False, pooling='avg')(inputs)
    outputs = Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_estimator(epochs=50, batch_size=64, steps_per_epoch=None, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
    # step 2. prepare model
    model = fe.build(model_def=DenseNet121_cifar10,
                     model_name="densenet121",
                     optimizer=tf.optimizers.Adam(lr=0.1),
                     loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss")
    ])
    # step 3.prepare estimator
    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        traces=[
            Accuracy(true_key="y", pred_key="y_pred"),
            ModelSaver(model_name="densenet121", save_dir=model_dir, save_best=True),
            LRController(model_name="densenet121", reduce_on_eval=True)
        ])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
