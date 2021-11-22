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

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, f1_score, roc_curve
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.backend import binary_crossentropy
from fastestimator.op.numpyop import LambdaOp
from fastestimator.op.numpyop.univariate import ExpandDims, Normalize
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.io import BestModelSaver
from fastestimator.util import to_number


def reconstructor(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    # Encoder Block
    model.add(
        layers.Conv2D(32, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(128, (5, 5),
                      strides=(2, 2),
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # Decoder Block
    model.add(
        layers.Conv2DTranspose(32, (5, 5),
                               strides=(2, 2),
                               output_padding=(0, 0),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Conv2DTranspose(16, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Conv2DTranspose(1, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                               activation='tanh'))
    return model


def discriminator(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(16, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                      input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(32, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(
        layers.Conv2D(128, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


class RLoss(TensorOp):
    def __init__(self, alpha=0.2, inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)
        self.alpha = alpha

    def forward(self, data, state):
        fake_score, x_fake, x = data
        recon_loss = binary_crossentropy(y_true=x, y_pred=x_fake, from_logits=True)
        adv_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.ones_like(fake_score), from_logits=True)
        return adv_loss + self.alpha * recon_loss


class DLoss(TensorOp):
    def forward(self, data, state):
        true_score, fake_score = data
        real_loss = binary_crossentropy(y_pred=true_score, y_true=tf.ones_like(true_score), from_logits=True)
        fake_loss = binary_crossentropy(y_pred=fake_score, y_true=tf.zeros_like(fake_score), from_logits=True)
        total_loss = real_loss + fake_loss
        return total_loss


class F1AUCScores(Trace):
    """Computes F1-Score and AUC Score for a classification task and reports it back to the logger.
    """
    def __init__(self, true_key, pred_key, mode=("eval", "test"), output_name=("auc_score", "f1_score")):
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode)
        self.y_true = []
        self.y_pred = []

    @property
    def true_key(self):
        return self.inputs[0]

    @property
    def pred_key(self):
        return self.inputs[1]

    def on_epoch_begin(self, data):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, data):
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        assert y_pred.size == y_true.size
        self.y_pred.extend(y_pred.ravel())
        self.y_true.extend(y_true.ravel())

    def on_epoch_end(self, data):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred, pos_label=1)  # (y, score, positive_label)
        roc_auc = auc(fpr, tpr)
        eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr - fpr)))]
        y_pred_class = np.copy(self.y_pred)
        y_pred_class[y_pred_class >= eer_threshold] = 1
        y_pred_class[y_pred_class < eer_threshold] = 0
        f_score = f1_score(self.y_true, y_pred_class, pos_label=0)

        data.write_with_log(self.outputs[0], roc_auc)
        data.write_with_log(self.outputs[1], f_score)


def get_estimator(epochs=20, batch_size=128, max_train_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    # Dataset Creation
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_eval0, y_eval0 = x_eval[np.where((y_eval == 1))], np.ones(y_eval[np.where((y_eval == 1))].shape)
    x_eval1, y_eval1 = x_eval[np.where((y_eval != 1))], y_eval[np.where((y_eval != 1))]

    # Ensuring outliers comprise 50% of the dataset
    index = np.random.choice(x_eval1.shape[0], int(x_eval0.shape[0]), replace=False)
    x_eval1, y_eval1 = x_eval1[index], np.zeros(y_eval1[index].shape)

    x_train, y_train = x_train[np.where((y_train == 1))], np.zeros(y_train[np.where((y_train == 1))].shape)
    train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})

    x_eval, y_eval = np.concatenate([x_eval0, x_eval1]), np.concatenate([y_eval0, y_eval1])
    eval_data = fe.dataset.NumpyDataset({"x": x_eval, "y": y_eval})

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x"),
            Normalize(inputs="x", outputs="x", mean=1.0, std=1.0, max_pixel_value=127.5),
            LambdaOp(fn=lambda x: x + np.random.normal(loc=0.0, scale=0.155, size=(28, 28, 1)),
                     inputs="x",
                     outputs="x_w_noise",
                     mode="train")
        ])

    recon_model = fe.build(model_fn=reconstructor,
                           optimizer_fn=lambda: tf.optimizers.RMSprop(2e-4),
                           model_name="reconstructor")
    disc_model = fe.build(model_fn=discriminator,
                          optimizer_fn=lambda: tf.optimizers.RMSprop(1e-4),
                          model_name="discriminator")

    network = fe.Network(ops=[
        ModelOp(model=recon_model, inputs="x_w_noise", outputs="x_fake", mode="train"),
        ModelOp(model=recon_model, inputs="x", outputs="x_fake", mode="eval"),
        ModelOp(model=disc_model, inputs="x_fake", outputs="fake_score"),
        ModelOp(model=disc_model, inputs="x", outputs="true_score"),
        RLoss(inputs=("fake_score", "x_fake", "x"), outputs="rloss"),
        UpdateOp(model=recon_model, loss_name="rloss"),
        DLoss(inputs=("true_score", "fake_score"), outputs="dloss"),
        UpdateOp(model=disc_model, loss_name="dloss"),
    ])

    traces = [
        F1AUCScores(true_key="y", pred_key="fake_score", mode="eval", output_name=["auc_score", "f1_score"]),
        BestModelSaver(model=recon_model, save_dir=save_dir, metric='f1_score', save_best_mode='max'),
        BestModelSaver(model=disc_model, save_dir=save_dir, metric='f1_score', save_best_mode='max'),
    ]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=max_train_steps_per_epoch,
                             log_steps=50)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
