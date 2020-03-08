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
"""DCGAN example using MNIST data set."""
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.backend import binary_crossentropy
from fastestimator.dataset import mnist
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ExpandDims, Normalize
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver


class GLoss(TensorOp):
    def forward(self, data, state):
        return binary_crossentropy(y_pred=data, y_true=tf.ones_like(data), from_logits=True)


class DLoss(TensorOp):
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = binary_crossentropy(y_pred=true_score, y_true=tf.ones_like(true_score), from_logits=True)
        fake_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss


def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


def discriminator():
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


def get_estimator(epochs=50, batch_size=256, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    train_data, _ = mnist.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x"),
            Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5),
            NumpyOp(inputs=lambda: np.random.normal(size=[100]).astype('float32'), outputs="z")
        ])
    gen_model = fe.build(model_fn=generator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    disc_model = fe.build(model_fn=discriminator, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(model=gen_model, inputs="z", outputs="x_fake"),
        ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
        GLoss(inputs="fake_score", outputs="gloss"),
        UpdateOp(model=gen_model, loss_name="gloss"),
        ModelOp(inputs="x", model=disc_model, outputs="true_score"),
        DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
        UpdateOp(model=disc_model, loss_name="dloss")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=ModelSaver(model=gen_model, save_dir=save_dir, frequency=5),
                             max_steps_per_epoch=max_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
