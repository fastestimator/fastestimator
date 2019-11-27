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
from tensorflow.keras import Model, layers

from fastestimator.layers import ApplyBias, EqualizedLRConv2D, EqualizedLRDense, FadeIn, MiniBatchStd, \
    PixelNormalization

fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
fmap_max = 512  # Maximum number of feature maps in any layer.


def _nf(stage):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


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
