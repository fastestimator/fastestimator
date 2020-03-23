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
"""
Convolutional Variational Auto Encoder (CVAE) example trained from MNIST dataset using Tensorflow backend
Ref: https://www.tensorflow.org/tutorials/generative/cvae
"""
import math
import tempfile
from typing import Any, Dict, Tuple

import tensorflow as tf

import fastestimator as fe
from fastestimator.dataset.data.mnist import load_data
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import Binarize, ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver

LATENT_DIM = 50


class SplitOp(TensorOp):
    """To split the infer net output into two """
    def forward(self, data: tf.Tensor, state: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
        mean, logvar = tf.split(data, num_or_size_splits=2, axis=1)
        return mean, logvar


class ReparameterizepOp(TensorOp):
    """Reparameterization trick. Ensures grads pass thru the sample to the infer net parameters"""
    def forward(self, data: Tuple[tf.Tensor, tf.Tensor], state: Dict[str, Any]) -> tf.Tensor:
        mean, logvar = data
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


class CVAELoss(TensorOp):
    """Convolutional variational auto-endcoder loss"""
    def forward(self, data: Tuple[tf.Tensor, ...], state: Dict[str, Any]) -> tf.Tensor:
        cross_ent, mean, logvar, z = data
        cross_ent = cross_ent * (28 * 28 * 1)
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        total_loss = cross_ent + tf.reduce_mean(-logpz + logqz_x)

        return total_loss

    @staticmethod
    def _log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * tf.constant(math.pi))
        return tf.reduce_sum(-.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def encoder_net():
    infer_model = tf.keras.Sequential()
    infer_model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    infer_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
    infer_model.add(tf.keras.layers.Flatten())
    infer_model.add(tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM))
    return infer_model


def decoder_net():
    generative_model = tf.keras.Sequential()
    generative_model.add(tf.keras.layers.InputLayer(input_shape=(LATENT_DIM, )))
    generative_model.add(tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu))
    generative_model.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
    generative_model.add(
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid'))
    return generative_model


def get_estimator(batch_size=100, epochs=100, max_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    train_data, test_data = load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=test_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x"),
            Minmax(inputs="x", outputs="x"),
            Binarize(inputs="x", outputs="x", threshold=0.5)
        ])

    encode_model = fe.build(model_fn=encoder_net, optimizer_fn="adam", model_names="encoder")
    decode_model = fe.build(model_fn=decoder_net, optimizer_fn="adam", model_names="decoder")

    network = fe.Network(ops=[
        ModelOp(model=encode_model, inputs="x", outputs="meanlogvar"),
        SplitOp(inputs="meanlogvar", outputs=("mean", "logvar")),
        ReparameterizepOp(inputs=("mean", "logvar"), outputs="z"),
        ModelOp(model=decode_model, inputs="z", outputs="x_logit"),
        CrossEntropy(inputs=("x_logit", "x"), outputs="cross_entropy"),
        CVAELoss(inputs=("cross_entropy", "mean", "logvar", "z"), outputs="loss"),
        UpdateOp(model=encode_model, loss_name="loss"),
        UpdateOp(model=decode_model, loss_name="loss"),
    ])

    traces = [
        BestModelSaver(model=encode_model, save_dir=save_dir), BestModelSaver(model=decode_model, save_dir=save_dir)
    ]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_steps_per_epoch=max_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
