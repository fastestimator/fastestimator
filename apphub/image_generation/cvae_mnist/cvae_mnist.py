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
"""Convolutional Variational Autoencoder example using MNIST data set."""

import math
import tempfile

import tensorflow as tf

import fastestimator as fe
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Loss, ModelOp
from fastestimator.trace import ModelSaver

LATENT_DIM = 50


def inference_net():
    infer_model = tf.keras.Sequential()
    infer_model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    infer_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Flatten())
    infer_model.add(tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM))
    return infer_model


def generative_net():
    generative_model = tf.keras.Sequential()
    generative_model.add(tf.keras.layers.InputLayer(input_shape=(LATENT_DIM, )))
    generative_model.add(tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu))
    generative_model.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME"))
    return generative_model


class Myrescale(TensorOp):
    """Normalize pixel values from uint8 to float32 between 0 and 1."""
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = data / 255
        return data


class Mybinarize(TensorOp):
    """Pixel values assigned to eithere 0 or 1 """
    def forward(self, data, state):
        data = tf.where(data >= 0.5, 1., 0.)
        return data


class SplitOp(TensorOp):
    """To split the infer net output into two """
    def forward(self, data, state):
        mean, logvar = tf.split(data, num_or_size_splits=2, axis=1)
        return mean, logvar


class ReparameterizepOp(TensorOp):
    """Reparameterization trick. Ensures grads pass thru the sample to the infer net parameters"""
    def forward(self, data, state):
        mean, logvar = data
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * tf.constant(math.pi))
    return tf.reduce_sum(-.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


class CVAELoss(Loss):
    """Convolutional variational auto-endcoder loss"""
    def forward(self, data, state):
        x, mean, logvar, z, x_logit = data
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = _log_normal_pdf(z, 0., 0.)
        logqz_x = _log_normal_pdf(z, mean, logvar)
        return -(logpx_z + logpz - logqz_x)


def get_estimator(batch_size=100, epochs=100, steps_per_epoch=None, model_dir=tempfile.mkdtemp()):
    # prepare data
    (x_train, _), (x_eval, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_eval = x_eval.reshape(x_eval.shape[0], 28, 28, 1).astype('float32')
    data = {"train": {"x": x_train}, "eval": {"x": x_eval}}
    pipeline = fe.Pipeline(batch_size=batch_size,
                           data=data,
                           ops=[Myrescale(inputs="x", outputs="x"), Mybinarize(inputs="x", outputs="x")])
    # prepare model
    infer_model = fe.build(model_def=inference_net,
                           model_name="encoder",
                           loss_name="loss",
                           optimizer=tf.optimizers.Adam(1e-4))
    gen_model = fe.build(model_def=generative_net,
                         model_name="decoder",
                         loss_name="loss",
                         optimizer=tf.optimizers.Adam(1e-4))

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=infer_model, outputs="meanlogvar", mode=None),
        SplitOp(inputs="meanlogvar", outputs=("mean", "logvar"), mode=None),
        ReparameterizepOp(inputs=("mean", "logvar"), outputs="z", mode=None),
        ModelOp(inputs="z", model=gen_model, outputs="x_logit"),
        CVAELoss(inputs=("x", "mean", "logvar", "z", "x_logit"), mode=None, outputs="loss")
    ])
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             traces=ModelSaver(model_name="decoder", save_dir=model_dir, save_best=True))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
