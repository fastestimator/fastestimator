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
"""This example showcase FastEstimator usage for tensorflow users. In this file, we use tf.dataset as data input.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, layers

import fastestimator as fe
from fastestimator.op.numpyop import Minmax, ExpandDims
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline, NumpyDataset
from fastestimator.trace.metric import Accuracy


def Scale(dataset):
    dataset["x"] = tf.cast(dataset["x"], tf.float32)
    dataset["y"] = tf.cast(dataset["y"], tf.int32)
    dataset["x"] = dataset["x"] / 255.0
    return dataset


def get_tensorflow_dataset(x, y, shuffle=True):
    data = {"x": x, "y": y}
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        data_length = x.shape[0]
        ds = ds.shuffle(data_length)
    ds = ds.map(Scale, num_parallel_calls=4)
    ds = ds.batch(32)
    ds = ds.prefetch(1)
    return ds


def LeNet(input_shape=(28, 28, 1), classes=10):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def get_estimator():
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

    # step 2
    model = fe.build(model=LeNet(), optimizer="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(
        pipeline={
            "train": get_tensorflow_dataset(x=np.expand_dims(x_train, -1), y=y_train),
            "eval": get_tensorflow_dataset(x=np.expand_dims(x_eval, -1), y=y_eval, shuffle=False)
        },
        network=network,
        epochs=2,
        traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
