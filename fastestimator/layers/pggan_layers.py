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
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers

fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
fmap_max = 512  # Maximum number of feature maps in any layer.


def nf(stage):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class FadeIn(layers.Add):
    def __init__(self, fade_in_alpha, **kwargs):
        super().__init__(**kwargs)
        self.fade_in_alpha = fade_in_alpha

    def get_config(self):
        return {'fade_in_alpha': self.fade_in_alpha}

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = ((1.0 - self.fade_in_alpha) * inputs[0]) + (self.fade_in_alpha * inputs[1])
        return output


class PixelNormalization(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def get_config(self):
        return {'eps': self.eps}

    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


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
        y = tf.reduce_mean(tf.square(y), axis=0)  #[MHWC]
        y = tf.sqrt(y + 1e-8)  # [MHWC]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.tile(y, [group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([x, y], axis=-1)


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
        fan_in = np.prod(input_shape[-1])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, x):
        return tf.matmul(x, self.w) * self.wscale


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


class ApplyBias(layers.Layer):
    def build(self, input_shape):
        self.b = self.add_weight(shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, x):
        return x + self.b
