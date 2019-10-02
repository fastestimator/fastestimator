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
from tensorflow.python.keras import Model, layers
from tensorflow.python.keras.initializers import RandomNormal
from fastestimator.layers import InstanceNormalization, ReflectionPadding2D


def _resblock(x0, num_filter=256, kernel_size=3):
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
    """Returns the discriminator network of the GAN.
    
    Args:
        input_shape (tuple, optional): shape of the input image. Defaults to (256, 256, 3).
    
    Returns:
        'Model' object: GAN discriminator.
    """
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
    """Returns the generator of the GAN.
    
    Args:
        input_shape (tuple, optional): shape of the input image. Defaults to (256, 256, 3).
        num_blocks (int, optional): number of resblocks for the generator. Defaults to 9.
    
    Returns:
        'Model' object: GAN generator.
    """
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
