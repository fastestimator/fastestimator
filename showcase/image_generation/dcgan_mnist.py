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
from tensorflow.python.keras import layers

from fastestimator.estimator.estimator import Estimator
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.util.op import TensorOp


class g_loss(Loss):
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def calculate_loss(self, batch, state):
        return self.cross_entropy(tf.ones_like(batch["pred_fake"]), batch["pred_fake"])


class d_loss(Loss):
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def calculate_loss(self, batch, state):
        real_loss = self.cross_entropy(tf.ones_like(batch["pred_true"]), batch["pred_true"])
        fake_loss = self.cross_entropy(tf.zeros_like(batch["pred_fake"]), batch["pred_fake"])
        total_loss = real_loss + fake_loss
        return total_loss


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


class Myrescale(TensorOp):
    def forward(self, data):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


def get_estimator():
    # prepare data
    (x_train, _), (x_eval, _) = tf.keras.datasets.mnist.load_data()
    data = {"train": {"x": np.expand_dims(x_train, -1)}, "eval": {"x": np.expand_dims(x_eval, -1)}}
    pipeline = Pipeline(batch_size=32,
                        data=data,
                        ops=Myrescale(inputs="x", outputs="x"))
    # prepare model
    g = build(keras_model=make_generator_model(), loss=g_loss(), optimizer=tf.optimizers.Adam(1e-4))
    d = build(keras_model=make_discriminator_model(), loss=d_loss(), optimizer=tf.optimizers.Adam(1e-4))
    network = Network(
        ops=[ModelOp(inputs=lambda: tf.random.normal([32, 100]), model=g), ModelOp(model=d, outputs="pred_fake"),
             ModelOp(inputs="x", model=d, outputs="pred_true")])
    # prepare estimator
    estimator = Estimator(network=network,
                          pipeline=pipeline,
                          epochs=2)
    return estimator
