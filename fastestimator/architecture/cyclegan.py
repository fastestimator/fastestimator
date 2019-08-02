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
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers
from tensorflow.keras.initializers import RandomNormal

# Code borrowed from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

# Code borrowd from https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def _resblock(x0, num_filter=256, kernel_size=3):
    x = ReflectionPadding2D()(x0)
    x = layers.Conv2D(filters=num_filter,
                    kernel_size=kernel_size,
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters=num_filter,
                    kernel_size=kernel_size, 
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = layers.Add()([x, x0])
    return x

def _build_discriminator(input_shape=(256, 256, 3)):
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
    x = layers.Conv2D(filters=512,
                    kernel_size=4,
                    strides=1,
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters=1,
                    kernel_size=4,
                    strides=1,
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

    return Model(inputs=x0, outputs=x)

def _build_generator(input_shape=(256, 256, 3), num_blocks=9):
    x0 = layers.Input(input_shape)

    x = ReflectionPadding2D(padding=(3, 3))(x0)
    x = layers.Conv2D(filters=64,
                    kernel_size=7,
                    strides=1,
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)

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
    x = layers.Conv2D(filters=3,
                    kernel_size=7,
                    activation='tanh',
                    kernel_initializer=RandomNormal(mean=0, stddev=0.02))(x)
    
    return Model(inputs=x0, outputs=x)
