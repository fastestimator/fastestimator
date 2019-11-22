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
from pathlib import Path

import numpy as np

import cv2
import fastestimator as fe
import tensorflow as tf
from fastestimator import RecordWriter
from fastestimator.architecture.pggan import build_D, build_G
from fastestimator.dataset.nih_chestxray import load_data
from fastestimator.op import TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.numpyop import Resize as ResizeRecord
from fastestimator.op.tensorop import Loss, ModelOp, Resize
from fastestimator.schedule import Scheduler
from fastestimator.trace import Trace
from tensorflow.python.keras import backend


class Rescale(TensorOp):
    """Scale image values from uint8 to float32 between -1 and 1."""
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


class CreateLowRes(TensorOp):
    def forward(self, data, state):
        data_shape = tf.shape(data)
        height = data_shape[0]
        width = data_shape[1]
        data = tf.image.resize(data, (height / 2, width / 2))
        data = tf.image.resize(data, (height, width))
        return data


class RandomInput(TensorOp):
    def forward(self, data, state):
        latent_dim = data
        batch_size = state["local_batch_size"]
        random_vector = tf.random.normal([batch_size, latent_dim])
        return random_vector


class ImageBlender(TensorOp):
    def __init__(self, alpha, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.alpha = alpha

    def forward(self, data, state):
        image, image_lowres = data
        new_img = self.alpha * image + (1 - self.alpha) * image_lowres
        return new_img


class Interpolate(TensorOp):
    def forward(self, data, state):
        fake, real = data
        batch_size = state["local_batch_size"]
        coeff = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
        return real + (fake - real) * coeff


class GradientPenalty(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        x_interp, interp_score = data
        interp_score = tf.reshape(interp_score, [-1])
        tape = state['tape']
        gradient_x_interp = tape.gradient(tf.reduce_sum(interp_score), x_interp)
        grad_l2 = tf.math.sqrt(tf.reduce_sum(tf.math.square(gradient_x_interp), axis=[1, 2, 3]))
        gp = tf.math.square(grad_l2 - 1.0)
        return gp


class GLoss(Loss):
    def forward(self, data, state):
        return -data


class DLoss(Loss):
    """Compute discriminator loss."""
    def __init__(self, inputs, outputs=None, mode=None, wgan_lambda=10, wgan_epsilon=0.001):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.wgan_lambda = wgan_lambda
        self.wgan_epsilon = wgan_epsilon

    def forward(self, data, state):
        real_score, fake_score, gp = data
        loss = fake_score - real_score + self.wgan_lambda * gp + tf.math.square(real_score) * self.wgan_epsilon
        return loss


class AlphaController(Trace):
    def __init__(self, alpha, fade_start, duration):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.alpha = alpha
        self.fade_start = fade_start
        self.duration = duration
        self.change_alpha = False
        self._idx = 0

    def on_epoch_begin(self, state):
        # check whetehr the current epoch is in smooth transition of resolutions
        fade_epoch = self.fade_start[self._idx]
        if state["epoch"] == fade_epoch:
            self.nimg_total = self.duration[self._idx] * state["num_examples"]
            self.change_alpha = True
            self.nimg_so_far = 0
            print("FastEstimator-Alpha: Started fading in for size {}".format(2**(self._idx + 3)))
        elif state["epoch"] == fade_epoch + self.duration[self._idx]:
            print("FastEstimator-Alpha: Finished fading in for size {}".format(2**(self._idx + 3)))
            self.change_alpha = False
            self._idx += 1
            backend.set_value(self.alpha, 1.0)

    def on_batch_begin(self, state):
        # if in resolution transition, smoothly change the alpha from 0 to 1
        if self.change_alpha:
            self.nimg_so_far += state["batch_size"]
            current_alpha = np.float32(self.nimg_so_far / self.nimg_total)
            backend.set_value(self.alpha, current_alpha)


class ImageSaving(Trace):
    def __init__(self, epoch_model, save_dir, num_sample=16, latent_dim=512, num_channels=3):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.epoch_model = epoch_model
        self.save_dir = save_dir
        self.latent_dim = latent_dim
        self.num_sample = num_sample
        self.num_channels = num_channels
        self.eps = 1e-8

    def on_epoch_end(self, state):
        if state["epoch"] in self.epoch_model:
            model = self.epoch_model[state["epoch"]]
            for i in range(self.num_sample):
                random_vectors = tf.random.normal([1, self.latent_dim])
                pred = model(random_vectors)
                disp_img = pred.numpy()
                disp_img = np.squeeze(disp_img)
                disp_img -= disp_img.min()
                disp_img /= (disp_img.max() + self.eps)
                disp_img = np.uint8(disp_img * 255)
                cv2.imwrite(os.path.join(self.save_dir, 'image_at_{:08d}_{}.png').format(state["epoch"], i), disp_img)
            print("on epoch {}, saving image to {}".format(state["epoch"], self.save_dir))


class ModelSaving(Trace):
    def __init__(self, epoch_model, save_dir):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.epoch_model = epoch_model
        self.save_dir = save_dir

    def on_epoch_end(self, state):
        if state["epoch"] in self.epoch_model:
            model = self.epoch_model[state["epoch"]]
            save_path = os.path.join(self.save_dir, model.model_name + ".h5")
            model.save(save_path, include_optimizer=False)
            print("FastEstimator-ModelSaver: Saving model to {}".format(save_path))


class ResetOptimizer(Trace):
    def __init__(self, reset_epochs, optimizer):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.reset_epochs = reset_epochs
        self.optimizer = optimizer

    def on_epoch_begin(self, state):
        if state["epoch"] in self.reset_epochs:
            for weight in self.optimizer.weights:
                backend.set_value(weight, weight - weight)
            print("Resetting optimizer on epoch {}".format(state["epoch"]))


def get_estimator(data_dir=None, save_dir=None):
    train_csv, data_path = load_data(data_dir)

    imreader = ImageReader(inputs="x", parent_path=data_path, grey_scale=True)
    writer_128 = RecordWriter(save_dir=os.path.join(data_path, "tfrecord_128"),
                              train_data=train_csv,
                              ops=[imreader, ResizeRecord(target_size=(128, 128), outputs="x")])
    writer_1024 = RecordWriter(save_dir=os.path.join(data_path, "tfrecord_1024"),
                               train_data=train_csv,
                               ops=[imreader, ResizeRecord(target_size=(1024, 1024), outputs="x")])
    # We create a scheduler for batch_size with the epochs at which it will change and corresponding values.
    batchsize_scheduler_128 = Scheduler({0: 128, 5: 64, 15: 32, 25: 16, 35: 8, 45: 4})
    batchsize_scheduler_1024 = Scheduler({55: 4, 65: 2, 75: 1})
    # pipeline ops
    resize_scheduler_128 = Scheduler({
        0: Resize(inputs="x", size=(4, 4), outputs="x"),
        5: Resize(inputs="x", size=(8, 8), outputs="x"),
        15: Resize(inputs="x", size=(16, 16), outputs="x"),
        25: Resize(inputs="x", size=(32, 32), outputs="x"),
        35: Resize(inputs="x", size=(64, 64), outputs="x"),
        45: None
    })
    resize_scheduler_1024 = Scheduler({
        55: Resize(inputs="x", size=(256, 256), outputs="x"),
        65: Resize(inputs="x", size=(512, 512), outputs="x"),
        75: None
    })
    lowres_op = CreateLowRes(inputs="x", outputs="x_lowres")
    rescale_x = Rescale(inputs="x", outputs="x")
    rescale_lowres = Rescale(inputs="x_lowres", outputs="x_lowres")
    pipeline_128 = fe.Pipeline(batch_size=batchsize_scheduler_128,
                               data=writer_128,
                               ops=[resize_scheduler_128, lowres_op, rescale_x, rescale_lowres])
    pipeline_1024 = fe.Pipeline(batch_size=batchsize_scheduler_1024,
                                data=writer_1024,
                                ops=[resize_scheduler_1024, lowres_op, rescale_x, rescale_lowres])

    pipeline_scheduler = Scheduler({0: pipeline_128, 55: pipeline_1024})

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

    fade_in_alpha = tf.Variable(initial_value=1.0, dtype='float32', trainable=False)

    d2, d3, d4, d5, d6, d7, d8, d9, d10 = fe.build(
        model_def=lambda: build_D(fade_in_alpha=fade_in_alpha, target_resolution=10, num_channels=1),
        model_name=["d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"],
        optimizer=[optimizer] * 9,
        loss_name=["dloss"] * 9)

    g2, g3, g4, g5, g6, g7, g8, g9, g10, G = fe.build(
        model_def=lambda: build_G(fade_in_alpha=fade_in_alpha, target_resolution=10, num_channels=1),
        model_name=["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "G"],
        optimizer=[optimizer] * 10,
        loss_name=["gloss"] * 10)

    g_scheduler = Scheduler({
        0: ModelOp(model=g2, outputs="x_fake"),
        5: ModelOp(model=g3, outputs="x_fake"),
        15: ModelOp(model=g4, outputs="x_fake"),
        25: ModelOp(model=g5, outputs="x_fake"),
        35: ModelOp(model=g6, outputs="x_fake"),
        45: ModelOp(model=g7, outputs="x_fake"),
        55: ModelOp(model=g8, outputs="x_fake"),
        65: ModelOp(model=g9, outputs="x_fake"),
        75: ModelOp(model=g10, outputs="x_fake")
    })

    fake_score_scheduler = Scheduler({
        0: ModelOp(inputs="x_fake", model=d2, outputs="fake_score"),
        5: ModelOp(inputs="x_fake", model=d3, outputs="fake_score"),
        15: ModelOp(inputs="x_fake", model=d4, outputs="fake_score"),
        25: ModelOp(inputs="x_fake", model=d5, outputs="fake_score"),
        35: ModelOp(inputs="x_fake", model=d6, outputs="fake_score"),
        45: ModelOp(inputs="x_fake", model=d7, outputs="fake_score"),
        55: ModelOp(inputs="x_fake", model=d8, outputs="fake_score"),
        65: ModelOp(inputs="x_fake", model=d9, outputs="fake_score"),
        75: ModelOp(inputs="x_fake", model=d10, outputs="fake_score")
    })

    real_score_scheduler = Scheduler({
        0: ModelOp(model=d2, outputs="real_score"),
        5: ModelOp(model=d3, outputs="real_score"),
        15: ModelOp(model=d4, outputs="real_score"),
        25: ModelOp(model=d5, outputs="real_score"),
        35: ModelOp(model=d6, outputs="real_score"),
        45: ModelOp(model=d7, outputs="real_score"),
        55: ModelOp(model=d8, outputs="real_score"),
        65: ModelOp(model=d9, outputs="real_score"),
        75: ModelOp(model=d10, outputs="real_score")
    })

    interp_score_scheduler = Scheduler({
        0:
        ModelOp(inputs="x_interp", model=d2, outputs="interp_score", track_input=True),
        5:
        ModelOp(inputs="x_interp", model=d3, outputs="interp_score", track_input=True),
        15:
        ModelOp(inputs="x_interp", model=d4, outputs="interp_score", track_input=True),
        25:
        ModelOp(inputs="x_interp", model=d5, outputs="interp_score", track_input=True),
        35:
        ModelOp(inputs="x_interp", model=d6, outputs="interp_score", track_input=True),
        45:
        ModelOp(inputs="x_interp", model=d7, outputs="interp_score", track_input=True),
        55:
        ModelOp(inputs="x_interp", model=d8, outputs="interp_score", track_input=True),
        65:
        ModelOp(inputs="x_interp", model=d9, outputs="interp_score", track_input=True),
        75:
        ModelOp(inputs="x_interp", model=d10, outputs="interp_score", track_input=True)
    })

    network = fe.Network(ops=[
        RandomInput(inputs=lambda: 512),
        g_scheduler,
        fake_score_scheduler,
        ImageBlender(inputs=("x", "x_lowres"), alpha=fade_in_alpha),
        real_score_scheduler,
        Interpolate(inputs=("x_fake", "x"), outputs="x_interp"),
        interp_score_scheduler,
        GradientPenalty(inputs=("x_interp", "interp_score"), outputs="gp"),
        GLoss(inputs="fake_score", outputs="gloss"),
        DLoss(inputs=("real_score", "fake_score", "gp"), outputs="dloss")
    ])

    if save_dir is None:
        save_dir = os.path.join(str(Path.home()), 'fastestimator_results', 'NIH_CXR_PGGAN')
        os.makedirs(save_dir, exist_ok=True)

    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline_scheduler,
        epochs=85,
        traces=[
            AlphaController(alpha=fade_in_alpha,
                            fade_start=[5, 15, 25, 35, 45, 55, 65, 75, 85],
                            duration=[5, 5, 5, 5, 5, 5, 5, 5, 5]),
            ResetOptimizer(reset_epochs=[5, 15, 25, 35, 45, 55, 65, 75], optimizer=optimizer),
            ImageSaving(epoch_model={
                4: g2, 14: g3, 24: g4, 34: g5, 44: g6, 54: g7, 64: g8, 74: g9, 84: G
            },
                        save_dir=save_dir,
                        num_channels=1),
            ModelSaving(epoch_model={84: G}, save_dir=save_dir)
        ])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
