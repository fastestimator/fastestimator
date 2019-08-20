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
import os
from glob import glob

import tensorflow as tf

from fastestimator.architecture.cyclegan import build_generator, build_discriminator
from fastestimator.dataset.zebra2horse_data import load_data
from fastestimator.record.record import RecordWriter
from fastestimator.record.preprocess import ImageReader
from fastestimator.estimator.trace import Trace
from fastestimator.estimator.estimator import Estimator
from fastestimator.network.loss import Loss
from fastestimator.network.model import ModelOp, build
from fastestimator.network.network import Network
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.util.op import TensorOp


class GifGenerator(Trace):
    def __init__(self, save_path, export_name="anim.gif"):
        super().__init__()
        self.save_path = save_path
        self.prefix = "image_at_epoch_{0:04d}.png"
        self.export_name = os.path.join(self.save_path, export_name)

    def begin(self, mode):
        if mode == "eval":
            if not (os.path.exists(self.save_path)):
                os.makedirs(self.save_path)

    def on_batch_end(self, state):
        import cv2
        mode = state['mode']
        epoch = state['epoch']

        if mode == "eval":
            img = state['batch']['prediction']['Y_fake']
            img = img[0, ...].numpy()
            img += 1
            img /= 2
            img *= 255
            img_path = os.path.join(self.save_path, self.prefix.format(epoch))
            cv2.imwrite(img_path, img.astype("uint8"))

    def end(self, mode):
        import imageio
        with imageio.get_writer(self.export_name, mode='I') as writer:
            filenames = glob(os.path.join(self.save_path, "*.png"))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 5 * (i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
                image = imageio.imread(filename)
                writer.append_data(image)


class Myrescale(TensorOp):
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


class RandomJitter(TensorOp):
    def forward(self, data, state):
        # resizing to 286 x 286 x 3
        data = tf.image.resize(data, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        data = tf.image.random_crop(data, size=[256, 256, 3])

        # random mirroring
        data = tf.image.random_flip_left_right(data)

        return data


class GLoss(Loss):
    def __init__(self, inputs, weight, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = weight

    def _adversarial_loss(self, fake_img):
        return self.cross_entropy(tf.ones_like(fake_img), fake_img)

    def _identity_loss(self, real_img, same_img):
        return 0.5 * self.LAMBDA * tf.reduce_mean(tf.abs(real_img - same_img))

    def _cycle_loss(self, real_img, cycled_img):
        return self.LAMBDA * tf.reduce_mean(tf.abs(real_img - cycled_img))

    def forward(self, data, state):
        real_img, fake_img, cycled_img, same_img = data
        total_loss = self._adversarial_loss(fake_img) + self._identity_loss(real_img, same_img) + self._cycle_loss(
            real_img, cycled_img)
        return total_loss


class DLoss(Loss):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def forward(self, data, state):
        real_img, fake_img = data
        real_img_loss = self.cross_entropy(tf.ones_like(real_img), real_img)
        fake_img_loss = self.cross_entropy(tf.zeros_like(real_img), fake_img)
        total_loss = real_img_loss + fake_img_loss
        return 0.5 * total_loss


def get_estimator(epochs=200):
    trainA_csv, trainB_csv, testA_csv, testB_csv, parent_path = load_data()

    # Step 1: Define Pipeline
    writer = RecordWriter(
        train_data=(trainA_csv, trainB_csv),
        ops=(ImageReader(inputs="imgA", outputs="imgA", parent_path=parent_path),
             ImageReader(inputs="imgB", outputs="imgB", parent_path=parent_path)))

    pipeline = Pipeline(
        data=writer,
        batch_size=1,
        ops=[
            Myrescale(inputs="imgA", outputs="imgA"),
            RandomJitter(inputs="imgA", outputs="real_A"),
            Myrescale(inputs="imgB", outputs="imgB"),
            RandomJitter(inputs="imgB", outputs="real_B")
        ])
    # Step2: Define Network
    g_AtoB = build(keras_model=build_generator(),
                   loss=GLoss(inputs=("real_A", "D_fake_B", "cycled_A", "same_A"), weight=10.0),
                   optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))
    g_BtoA = build(keras_model=build_generator(),
                   loss=GLoss(inputs=("real_B", "D_fake_A", "cycled_B", "same_B"), weight=10.0),
                   optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))
    d_A = build(keras_model=build_discriminator(),
                loss=DLoss(inputs=("D_real_A", "D_fake_A")),
                optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))
    d_B = build(keras_model=build_discriminator(),
                loss=DLoss(inputs=("D_real_B", "D_fake_B")),
                optimizer=tf.keras.optimizers.Adam(2e-4, 0.5))

    network = Network(ops=[
        ModelOp(inputs="real_A", model=g_AtoB, outputs="fake_B"),
        ModelOp(inputs="real_A", model=d_A, outputs="D_real_A"),
        ModelOp(inputs="real_A", model=g_BtoA, outputs="same_A"),
        ModelOp(inputs="real_B", model=g_BtoA, outputs="fake_A"),
        ModelOp(inputs="real_B", model=d_B, outputs="D_real_B"),
        ModelOp(inputs="real_B", model=g_AtoB, outputs="same_B"),
        ModelOp(inputs="fake_B", model=g_BtoA, outputs="cycled_A"),
        ModelOp(inputs="fake_B", model=d_B, outputs="D_fake_B"),
        ModelOp(inputs="fake_A", model=g_AtoB, outputs="cycled_B"),
        ModelOp(inputs="fake_A", model=d_A, outputs="D_fake_A")
    ])
    # Step3: Define Estimator
    # traces = [GifGenerator("/root/data/public/horse2zebra/images")]
    estimator = Estimator(network=network, pipeline=pipeline, epochs=epochs)
    return estimator
