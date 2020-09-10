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
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, backend, layers
from tensorflow.keras.optimizers import Adam

import fastestimator as fe
from fastestimator.backend import feed_forward, get_gradient
from fastestimator.dataset.data import nih_chestxray
from fastestimator.op.numpyop import LambdaOp
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler
from fastestimator.trace import Trace
from fastestimator.trace.io import ModelSaver
from fastestimator.util import get_num_devices


def _nf(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class EqualizedLRDense(layers.Layer):
    def __init__(self, units, gain=np.sqrt(2)):
        super().__init__()
        self.units = units
        self.gain = gain

    def get_config(self):
        return {'units': self.units, 'gain': self.gain}

    def build(self, input_shape):
        self.w = self.add_weight(shape=[int(input_shape[-1]), self.units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
                                 trainable=True)
        fan_in = input_shape[-1]
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, x):
        return tf.matmul(x, self.w) * self.wscale


class ApplyBias(layers.Layer):
    def build(self, input_shape):
        self.b = self.add_weight(shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, x):
        return x + self.b


class EqualizedLRConv2D(layers.Conv2D):
    def __init__(self, filters, gain=np.sqrt(2), kernel_size=3, strides=(1, 1), padding="same"):
        super().__init__(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.float32(np.prod(self.kernel.shape[:-1]))
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def get_config(self):
        return {'filters': self.filters, 'gain': self.gain, "kernel_size": self.kernel_size}

    def call(self, x):
        return super().call(x) * self.wscale


class PixelNormalization(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def get_config(self):
        return {'eps': self.eps}

    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


def _block_G(res, latent_dim=512, initial_resolution=2):
    if res == initial_resolution:
        x0 = layers.Input(shape=latent_dim)
        x = PixelNormalization()(x0)

        x = EqualizedLRDense(units=_nf(res - 1) * 16, gain=np.sqrt(2) / 4)(x)
        x = tf.reshape(x, [-1, 4, 4, _nf(res - 1)])
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)

        x = EqualizedLRConv2D(filters=_nf(res - 1))(x)
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)
    else:
        x0 = layers.Input(shape=(2**(res - 1), 2**(res - 1), _nf(res - 2)))
        x = layers.UpSampling2D()(x0)
        for _ in range(2):
            x = EqualizedLRConv2D(filters=_nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = PixelNormalization()(x)
    return Model(inputs=x0, outputs=x, name="g_block_%dx%d" % (2**res, 2**res))


class MiniBatchStd(layers.Layer):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def get_config(self):
        return {'group_size': self.group_size}

    def call(self, x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = x.shape  # [NHWC]
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC]
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MHWC]
        y = tf.sqrt(y + 1e-8)  # [MHWC]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.tile(y, [group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([x, y], axis=-1)


class FadeIn(layers.Add):
    def __init__(self, fade_in_alpha, **kwargs):
        super().__init__(**kwargs)
        self.fade_in_alpha = fade_in_alpha

    def get_config(self):
        return {'fade_in_alpha': self.fade_in_alpha}

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = (1.0 - self.fade_in_alpha) * inputs[0] + self.fade_in_alpha * inputs[1]
        return output


def _torgb(res, num_channels=3):
    x0 = layers.Input(shape=(2**res, 2**res, _nf(res - 1)))
    x = EqualizedLRConv2D(filters=num_channels, kernel_size=1, gain=1.0)(x0)
    x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="to_rgb_%dx%d" % (2**res, 2**res))


def build_G(fade_in_alpha, latent_dim=512, initial_resolution=2, target_resolution=10, num_channels=3):
    x0 = layers.Input(shape=latent_dim)
    curr_g_block = _block_G(initial_resolution, latent_dim, initial_resolution)
    curr_to_rgb_block = _torgb(initial_resolution, num_channels)
    images_out = curr_g_block(x0)
    images_out = curr_to_rgb_block(images_out)
    model_list = list()
    gen_block_list = list()
    mdl = Model(inputs=x0, outputs=images_out)
    model_list.append(mdl)
    gen_block_list.append(curr_g_block)
    prev_to_rgb_block = curr_to_rgb_block
    for res in range(initial_resolution + 1, target_resolution + 1):
        curr_g_block = _block_G(res, latent_dim, initial_resolution)
        curr_to_rgb_block = _torgb(res, num_channels)
        prev_images = x0
        for g in gen_block_list:
            prev_images = g(prev_images)
        curr_images = curr_g_block(prev_images)
        curr_images = curr_to_rgb_block(curr_images)
        prev_images = prev_to_rgb_block(prev_images)
        prev_images = layers.UpSampling2D(name="upsample_%dx%d" % (2**res, 2**res))(prev_images)
        images_out = FadeIn(fade_in_alpha=fade_in_alpha,
                            name="fade_in_%dx%d" % (2**res, 2**res))([prev_images, curr_images])
        mdl = Model(inputs=x0, outputs=images_out)
        model_list.append(mdl)
        gen_block_list.append(curr_g_block)
        prev_to_rgb_block = curr_to_rgb_block
    # build final model
    x = x0
    for g in gen_block_list:
        x = g(x)
    x = curr_to_rgb_block(x)
    final_mdl = Model(inputs=x0, outputs=x)
    model_list.append(final_mdl)
    return model_list


def _fromrgb(res, num_channels=3):
    x0 = layers.Input(shape=(2**res, 2**res, num_channels))
    x = EqualizedLRConv2D(filters=_nf(res - 1), kernel_size=1)(x0)
    x = ApplyBias()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return Model(inputs=x0, outputs=x, name="from_rgb_%dx%d" % (2**res, 2**res))


def _block_D(res, initial_resolution, mbstd_group_size=4):
    x0 = layers.Input(shape=(2**res, 2**res, _nf(res - 1)))
    if res > initial_resolution:
        x = x0
        for i in range(2):
            x = EqualizedLRConv2D(filters=_nf(res - (i + 1)))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.AveragePooling2D()(x)
    else:
        if mbstd_group_size > 1:
            x = MiniBatchStd(mbstd_group_size)(x0)
            x = EqualizedLRConv2D(filters=_nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            x = layers.Flatten()(x)
            x = EqualizedLRDense(units=_nf(res - 2))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = EqualizedLRDense(units=1, gain=1.0)(x)
            x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="d_block_%dx%d" % (2**res, 2**res))


def build_D(fade_in_alpha, mbstd_group_size=4, initial_resolution=2, target_resolution=10, num_channels=3):
    model_list = list()
    disc_block_list = list()
    for res in range(initial_resolution, target_resolution + 1):
        x0 = layers.Input(shape=(2**res, 2**res, num_channels))
        curr_from_rgb = _fromrgb(res, num_channels)
        curr_D_block = _block_D(res, initial_resolution, mbstd_group_size)
        x = curr_from_rgb(x0)
        x = curr_D_block(x)
        if res > initial_resolution:
            x_ds = layers.AveragePooling2D(name="downsample_%dx%d" % (2**res, 2**res))(x0)
            x_ds = prev_from_rgb(x_ds)
            x = FadeIn(fade_in_alpha=fade_in_alpha, name="fade_in_%dx%d" % (2**res, 2**res))([x_ds, x])
            for prev_d in disc_block_list[::-1]:
                x = prev_d(x)
        disc_block_list.append(curr_D_block)
        prev_from_rgb = curr_from_rgb
        mdl = Model(inputs=x0, outputs=x)
        model_list.append(mdl)
    return model_list


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
        batch_size = real.shape[0]
        coeff = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
        return real + (fake - real) * coeff


class GradientPenalty(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        x_interp, interp_score = data
        gradient_x_interp = get_gradient(tf.reduce_sum(interp_score), x_interp, higher_order=True, tape=state['tape'])
        grad_l2 = tf.math.sqrt(tf.reduce_sum(tf.math.square(gradient_x_interp), axis=[1, 2, 3]))
        gp = tf.math.square(grad_l2 - 1.0)
        return gp


class GLoss(TensorOp):
    def forward(self, data, state):
        return -tf.reduce_mean(data)


class DLoss(TensorOp):
    """Compute discriminator loss."""
    def __init__(self, inputs, outputs=None, mode=None, wgan_lambda=10, wgan_epsilon=0.001):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.wgan_lambda = wgan_lambda
        self.wgan_epsilon = wgan_epsilon

    def forward(self, data, state):
        real_score, fake_score, gp = data
        loss = fake_score - real_score + self.wgan_lambda * gp + tf.math.square(real_score) * self.wgan_epsilon
        return tf.reduce_mean(loss)


class AlphaController(Trace):
    def __init__(self, alpha, fade_start_epochs, duration, batch_scheduler, num_examples):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.alpha = alpha
        self.fade_start_epochs = fade_start_epochs
        self.duration = duration
        self.batch_scheduler = batch_scheduler
        self.num_examples = num_examples
        self.change_alpha = False
        self.nimg_total = self.duration * self.num_examples
        self._idx = 0
        self.current_batch_size = None

    def on_epoch_begin(self, state):
        # check whether the current epoch is in smooth transition of resolutions
        fade_epoch = self.fade_start_epochs[self._idx]
        if self.system.epoch_idx == fade_epoch:
            self.change_alpha = True
            self.nimg_so_far = 0
            self.current_batch_size = self.batch_scheduler.get_current_value(self.system.epoch_idx)
            print("FastEstimator-Alpha: Started fading in for size {}".format(2**(self._idx + 3)))
        elif self.system.epoch_idx == fade_epoch + self.duration:
            print("FastEstimator-Alpha: Finished fading in for size {}".format(2**(self._idx + 3)))
            self.change_alpha = False
            if self._idx + 1 < len(self.fade_start_epochs):
                self._idx += 1
            backend.set_value(self.alpha, 1.0)

    def on_batch_begin(self, state):
        # if in resolution transition, smoothly change the alpha from 0 to 1
        if self.change_alpha:
            self.nimg_so_far += self.current_batch_size
            current_alpha = np.float32(self.nimg_so_far / self.nimg_total)
            backend.set_value(self.alpha, current_alpha)


class ImageSaving(Trace):
    def __init__(self, epoch_model_map, save_dir, num_sample=16, latent_dim=512):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.epoch_model_map = epoch_model_map
        self.save_dir = save_dir
        self.latent_dim = latent_dim
        self.num_sample = num_sample
        self.eps = 1e-8

    def on_epoch_end(self, state):
        if self.system.epoch_idx in self.epoch_model_map:
            model = self.epoch_model_map[self.system.epoch_idx]
            for i in range(self.num_sample):
                random_vectors = tf.random.normal([1, self.latent_dim])
                pred = feed_forward(model, random_vectors, training=False)
                disp_img = pred.numpy()
                disp_img = np.squeeze(disp_img)
                disp_img -= disp_img.min()
                disp_img /= (disp_img.max() + self.eps)
                disp_img = np.uint8(disp_img * 255)
                cv2.imwrite(
                    os.path.join(self.save_dir, 'image_at_{:08d}_{}.png').format(self.system.epoch_idx, i), disp_img)
            print("on epoch {}, saving image to {}".format(self.system.epoch_idx, self.save_dir))


def get_estimator(target_size=128,
                  epochs=55,
                  save_dir=tempfile.mkdtemp(),
                  max_train_steps_per_epoch=None,
                  data_dir=None):
    # assert growth parameters
    num_grow = np.log2(target_size) - 2
    assert num_grow >= 1 and num_grow % 1 == 0, "need exponential of 2 and greater than 8 as target size"
    num_phases = int(2 * num_grow + 1)
    assert epochs % num_phases == 0, "epoch must be multiple of {} for size {}".format(num_phases, target_size)
    num_grow, phase_length = int(num_grow), int(epochs / num_phases)
    event_epoch = [1, 1 + phase_length] + [phase_length * (2 * i + 1) + 1 for i in range(1, num_grow)]
    event_size = [4] + [2**(i + 3) for i in range(num_grow)]
    # set up data schedules
    dataset = nih_chestxray.load_data(root_dir=data_dir)
    resize_map = {
        epoch: Resize(image_in="x", image_out="x", height=size, width=size)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    resize_low_res_map1 = {
        epoch: Resize(image_in="x", image_out="x_low_res", height=size // 2, width=size // 2)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    resize_low_res_map2 = {
        epoch: Resize(image_in="x_low_res", image_out="x_low_res", height=size, width=size)
        for (epoch, size) in zip(event_epoch, event_size)
    }
    batch_size_map = {
        epoch: 512 // size * get_num_devices() if size <= 128 else 4 * get_num_devices()
        for (epoch, size) in zip(event_epoch, event_size)
    }
    batch_scheduler = EpochScheduler(epoch_dict=batch_size_map)
    pipeline = fe.Pipeline(
        batch_size=batch_scheduler,
        train_data=dataset,
        drop_last=True,
        ops=[
            ReadImage(inputs="x", outputs="x", color_flag='gray'),
            EpochScheduler(epoch_dict=resize_map),
            EpochScheduler(epoch_dict=resize_low_res_map1),
            EpochScheduler(epoch_dict=resize_low_res_map2),
            Normalize(inputs=["x", "x_low_res"], outputs=["x", "x_low_res"], mean=1.0, std=1.0, max_pixel_value=127.5),
            LambdaOp(fn=lambda: np.random.normal(size=[512]).astype('float32'), outputs="z")
        ])
    # now model schedule
    fade_in_alpha = tf.Variable(initial_value=1.0, dtype='float32', trainable=False)
    d_models = fe.build(
        model_fn=lambda: build_D(fade_in_alpha, target_resolution=int(np.log2(target_size)), num_channels=1),
        optimizer_fn=[lambda: Adam(0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)] * len(event_size),
        model_name=["d_{}".format(size) for size in event_size])
    g_models = fe.build(
        model_fn=lambda: build_G(fade_in_alpha, target_resolution=int(np.log2(target_size)), num_channels=1),
        optimizer_fn=[lambda: Adam(0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)] * len(event_size) + [None],
        model_name=["g_{}".format(size) for size in event_size] + ["G"])
    fake_img_map = {
        epoch: ModelOp(inputs="z", outputs="x_fake", model=model)
        for (epoch, model) in zip(event_epoch, g_models[:-1])
    }
    fake_score_map = {
        epoch: ModelOp(inputs="x_fake", outputs="fake_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    real_score_map = {
        epoch: ModelOp(inputs="x_blend", outputs="real_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    interp_score_map = {
        epoch: ModelOp(inputs="x_interp", outputs="interp_score", model=model)
        for (epoch, model) in zip(event_epoch, d_models)
    }
    g_update_map = {
        epoch: UpdateOp(loss_name="gloss", model=model)
        for (epoch, model) in zip(event_epoch, g_models[:-1])
    }
    d_update_map = {epoch: UpdateOp(loss_name="dloss", model=model) for (epoch, model) in zip(event_epoch, d_models)}
    network = fe.Network(ops=[
        EpochScheduler(fake_img_map),
        EpochScheduler(fake_score_map),
        ImageBlender(alpha=fade_in_alpha, inputs=("x", "x_low_res"), outputs="x_blend"),
        EpochScheduler(real_score_map),
        Interpolate(inputs=("x_fake", "x"), outputs="x_interp"),
        EpochScheduler(interp_score_map),
        GradientPenalty(inputs=("x_interp", "interp_score"), outputs="gp"),
        GLoss(inputs="fake_score", outputs="gloss"),
        DLoss(inputs=("real_score", "fake_score", "gp"), outputs="dloss"),
        EpochScheduler(g_update_map),
        EpochScheduler(d_update_map)
    ])
    traces = [
        AlphaController(alpha=fade_in_alpha,
                        fade_start_epochs=event_epoch[1:],
                        duration=phase_length,
                        batch_scheduler=batch_scheduler,
                        num_examples=len(dataset)),
        ModelSaver(model=g_models[-1], save_dir=save_dir, frequency=phase_length),
        ImageSaving(
            epoch_model_map={epoch - 1: model
                             for (epoch, model) in zip(event_epoch[1:] + [epochs + 1], g_models[:-1])},
            save_dir=save_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
