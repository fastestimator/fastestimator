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
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import RandomNormal

import fastestimator as fe
from fastestimator.backend import reduce_mean
from fastestimator.dataset.data.horse2zebra import load_data
from fastestimator.layers.tensorflow import InstanceNormalization, ReflectionPadding2D
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, RandomCrop, Resize
from fastestimator.op.numpyop.univariate import Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import ModelSaver


def lr_schedule(epoch):
    """Learning rate schedule"""
    if epoch <= 100:
        lr = 2e-4
    else:
        lr = 2e-4 * (200 - epoch) / 100
    return lr


class PlaceholderOp(NumpyOp):
    """NumpyOp to generate dummy keys required by the Network"""
    def forward(self, data, state):
        return 1.0, np.zeros(shape=(256, 256, 3))


class Buffer(TensorOp):
    """Image Buffer implementation as outlined in https://arxiv.org/abs/1703.10593v6"""
    def __init__(self, image_in=None, buffer_in=None, index_in=None, image_out=None, mode=None):
        super().__init__(inputs=[image_in, buffer_in, index_in], outputs=image_out, mode=mode)

    def forward(self, data, state):
        image_in, buffer_in, index_in = data
        index_in = tf.reshape(index_in, shape=(-1, 1, 1, 1))
        index_in = tf.cast(index_in, dtype=tf.float32)
        buffer_in = tf.cast(buffer_in, dtype=tf.float32)
        output = tf.multiply(image_in, index_in) + buffer_in
        return output


class BufferUpdate(Trace):
    """Trace to update Image Buffer"""
    def __init__(self, input_name="fake", buffer_size=50, batch_size=1, mode="train", output_name=("buffer", "index")):
        super().__init__(inputs=input_name, mode=mode, outputs=output_name)
        self.input_key = input_name
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.image_buffer = []
        self.num_imgs = 0
        self.data_index = []
        self.buffer_index = []

    def on_batch_begin(self, data):
        self.data_index = []
        self.buffer_index = []
        buffer_fill = 0
        buffer_output = []
        for _ in range(self.batch_size):
            if self.num_imgs + buffer_fill < self.buffer_size:
                self.data_index.append(1.)
                self.buffer_index.append(-1)
                buffer_fill += 1
                buffer_output.append(tf.zeros(shape=(1, 256, 256, 3)))
            else:
                if np.random.uniform() > 0.5:
                    random_idx = np.random.randint(self.buffer_size)
                    self.buffer_index.append(random_idx)
                    buffer_output.append(self.image_buffer[random_idx])
                    self.data_index.append(0.)
                else:
                    self.data_index.append(1.)
                    self.buffer_index.append(-1)
                    buffer_output.append(tf.zeros(shape=(1, 256, 256, 3)))

        buffer_output = tf.concat(buffer_output, 0)

        data.write_without_log(self.outputs[0], buffer_output)
        data.write_without_log(self.outputs[1], self.data_index)

    def on_batch_end(self, data):
        fake_imgs = data[self.input_key]
        for i, image in enumerate(fake_imgs):
            image = tf.expand_dims(image, 0)
            if self.num_imgs < self.buffer_size:
                self.image_buffer.append(image)
                self.num_imgs += 1
            else:
                if self.buffer_index[i] != -1:
                    self.image_buffer[self.buffer_index[i]] = image


class GLoss(TensorOp):
    """TensorOp to compute generator loss"""
    def __init__(self, inputs, weight, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.LAMBDA = weight
        self.average_loss = average_loss

    def _adversarial_loss(self, fake_img):
        return tf.reduce_mean(self.loss_fn(tf.ones_like(fake_img), fake_img), axis=(1, 2))

    def _identity_loss(self, real_img, same_img):
        return 0.5 * self.LAMBDA * tf.reduce_mean(tf.abs(real_img - same_img), axis=(1, 2, 3))

    def _cycle_loss(self, real_img, cycled_img):
        return self.LAMBDA * tf.reduce_mean(tf.abs(real_img - cycled_img), axis=(1, 2, 3))

    def forward(self, data, state):
        real_img, fake_img, cycled_img, same_img = data
        total_loss = self._adversarial_loss(fake_img) + self._identity_loss(real_img, same_img) + self._cycle_loss(
            real_img, cycled_img)

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return total_loss


class DLoss(TensorOp):
    """TensorOp to compute discriminator loss"""
    def __init__(self, inputs, outputs=None, mode=None, average_loss=True):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.average_loss = average_loss

    def forward(self, data, state):
        real_img, fake_img = data
        real_img_loss = tf.reduce_mean(self.loss_fn(tf.ones_like(real_img), real_img), axis=(1, 2))
        fake_img_loss = tf.reduce_mean(self.loss_fn(tf.zeros_like(real_img), fake_img), axis=(1, 2))
        total_loss = real_img_loss + fake_img_loss

        if self.average_loss:
            total_loss = reduce_mean(total_loss)

        return 0.5 * total_loss


def _resblock(x0, num_filter=256, kernel_size=3):
    """Residual block architecture"""
    x = ReflectionPadding2D()(x0)
    x = layers.Conv2D(filters=num_filter, kernel_size=kernel_size, kernel_initializer=RandomNormal(mean=0,
                                                                                                   stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters=num_filter, kernel_size=kernel_size, kernel_initializer=RandomNormal(mean=0,
                                                                                                   stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.Add()([x, x0])
    return x


def build_discriminator(input_shape=(256, 256, 3)):
    """Discriminator network architecture"""
    x0 = layers.Input(input_shape)
    x = layers.Conv2D(filters=64,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x0)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters=512, kernel_size=4, strides=1, kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters=1, kernel_size=4, strides=1, kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    return Model(inputs=x0, outputs=x)


def build_generator(input_shape=(256, 256, 3), num_blocks=9):
    """Generator network architecture"""
    x0 = layers.Input(input_shape)

    x = ReflectionPadding2D(padding=(3, 3))(x0)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=1, kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # downsample
    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # residual
    for _ in range(num_blocks):
        x = _resblock(x)

    # upsample
    x = layers.Conv2DTranspose(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(filters=64,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # final
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(filters=3, kernel_size=7, activation='tanh', kernel_initializer=RandomNormal(mean=0,
                                                                                                   stddev=0.02))(x)

    return Model(inputs=x0, outputs=x)


def get_estimator(weight=10.0,
                  epochs=200,
                  batch_size=1,
                  train_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp(),
                  data_dir=None):
    train_data, _ = load_data(batch_size=batch_size, root_dir=data_dir)

    pipeline = fe.Pipeline(
        train_data=train_data,
        ops=[
            ReadImage(inputs=["A", "B"], outputs=["A", "B"]),
            Normalize(inputs=["A", "B"], outputs=["real_A", "real_B"], mean=1.0, std=1.0, max_pixel_value=127.5),
            Resize(height=286, width=286, image_in="real_A", image_out="real_A", mode="train"),
            RandomCrop(height=256, width=256, image_in="real_A", image_out="real_A", mode="train"),
            Resize(height=286, width=286, image_in="real_B", image_out="real_B", mode="train"),
            RandomCrop(height=256, width=256, image_in="real_B", image_out="real_B", mode="train"),
            Sometimes(HorizontalFlip(image_in="real_A", image_out="real_A", mode="train")),
            Sometimes(HorizontalFlip(image_in="real_B", image_out="real_B", mode="train")),
            PlaceholderOp(outputs=("index_A", "buffer_A")),
            PlaceholderOp(outputs=("index_B", "buffer_B"))
        ])

    g_AtoB = fe.build(model_fn=build_generator, model_name="g_AtoB", optimizer_fn=lambda: tf.optimizers.Adam(2e-4, 0.5))
    g_BtoA = fe.build(model_fn=build_generator, model_name="g_BtoA", optimizer_fn=lambda: tf.optimizers.Adam(2e-4, 0.5))
    d_A = fe.build(model_fn=build_discriminator, model_name="d_A", optimizer_fn=lambda: tf.optimizers.Adam(2e-4, 0.5))
    d_B = fe.build(model_fn=build_discriminator, model_name="d_B", optimizer_fn=lambda: tf.optimizers.Adam(2e-4, 0.5))

    network = fe.Network(ops=[
        ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
        ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
        Buffer(image_in="fake_A", buffer_in="buffer_A", index_in="index_A", image_out="buffer_fake_A"),
        Buffer(image_in="fake_B", buffer_in="buffer_B", index_in="index_B", image_out="buffer_fake_B"),
        ModelOp(inputs="real_A", model=d_A, outputs="d_real_A"),
        ModelOp(inputs="fake_A", model=d_A, outputs="d_fake_A"),
        ModelOp(inputs="buffer_fake_A", model=d_A, outputs="buffer_d_fake_A"),
        ModelOp(inputs="real_B", model=d_B, outputs="d_real_B"),
        ModelOp(inputs="fake_B", model=d_B, outputs="d_fake_B"),
        ModelOp(inputs="buffer_fake_B", model=d_B, outputs="buffer_d_fake_B"),
        ModelOp(inputs="real_A", model=g_BtoA, outputs="same_A"),
        ModelOp(inputs="fake_B", model=g_BtoA, outputs="cycled_A"),
        ModelOp(inputs="real_B", model=g_AtoB, outputs="same_B"),
        ModelOp(inputs="fake_A", model=g_AtoB, outputs="cycled_B"),
        GLoss(inputs=("real_A", "d_fake_B", "cycled_A", "same_A"), weight=weight, outputs="g_AtoB_loss"),
        GLoss(inputs=("real_B", "d_fake_A", "cycled_B", "same_B"), weight=weight, outputs="g_BtoA_loss"),
        DLoss(inputs=("d_real_A", "buffer_d_fake_A"), outputs="d_A_loss"),
        DLoss(inputs=("d_real_B", "buffer_d_fake_B"), outputs="d_B_loss"),
        UpdateOp(model=g_AtoB, loss_name="g_AtoB_loss"),
        UpdateOp(model=g_BtoA, loss_name="g_BtoA_loss"),
        UpdateOp(model=d_A, loss_name="d_A_loss"),
        UpdateOp(model=d_B, loss_name="d_B_loss")
    ])

    traces = [
        BufferUpdate(input_name="fake_A",
                     buffer_size=50,
                     batch_size=batch_size,
                     mode="train",
                     output_name=["buffer_A", "index_A"]),
        BufferUpdate(input_name="fake_B",
                     buffer_size=50,
                     batch_size=batch_size,
                     mode="train",
                     output_name=["buffer_B", "index_B"]),
        ModelSaver(model=g_AtoB, save_dir=save_dir, frequency=5),
        ModelSaver(model=g_BtoA, save_dir=save_dir, frequency=5),
        LRScheduler(model=g_AtoB, lr_fn=lr_schedule),
        LRScheduler(model=g_BtoA, lr_fn=lr_schedule),
        LRScheduler(model=d_A, lr_fn=lr_schedule),
        LRScheduler(model=d_B, lr_fn=lr_schedule)
    ]

    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
