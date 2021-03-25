# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers


# noinspection PyPep8Naming
def WideResidualNetwork(input_shape: Tuple[int, int, int],
                        depth: int = 28,
                        widen_factor: int = 10,
                        dropout: float = 0.0,
                        classes: int = 10,
                        activation: Optional[str] = 'softmax') -> tf.keras.Model:
    """Creates a Wide Residual Network with specified parameters.

    Args:
        input_shape: The size of the input tensor (height, width, channels).
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        widen_factor: Width of the network.
        dropout: Adds dropout if value is greater than 0.0.
        classes: The number of outputs the model should generate.
        activation: activation function for last dense layer.

    Returns:
        A Keras Model.

    Raises:
        ValueError: If (depth - 4) is not divisible by 6.
    """
    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)' 'should be divisible by 6.')

    img_input = layers.Input(shape=input_shape)
    inputs = img_input

    x = __create_wide_residual_network(classes, img_input, depth, widen_factor, dropout, activation)
    # Create model.
    model = Model(inputs, x)
    return model


def __conv1_block(inputs: tf.Tensor, n_filters: int) -> tf.Tensor:
    """Conv block of the network.

    Args:
        inputs: input tensor.
        n_filters: How many filters for the convolution layer.

    Returns:
        Output tensor of the conv block.
    """
    x = layers.Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation('relu')(x)
    return x


def __basic_block(inputs: tf.Tensor, n_filters: int = 16, dropout: float = 0.0) -> tf.Tensor:
    """Basic block of the network.

    Args:
        inputs: input tensor.
        n_filters: How many filters for the convolution layer.
        dropout: Adds dropout if value is greater than 0.0.

    Returns:
        Output tensor of the basic block.
    """
    init = inputs

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if init.shape[-1] != n_filters:
        init = layers.Conv2D(n_filters, (1, 1),
                             activation='linear',
                             padding='same',
                             kernel_initializer='he_normal',
                             use_bias=False)(init)

    x = layers.Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation('relu')(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation('relu')(x)

    m = init + x
    return m


def __create_wide_residual_network(classes: int,
                                   img_input: tf.Tensor,
                                   depth: int = 28,
                                   widen_factor: int = 10,
                                   dropout: float = 0.0,
                                   activation: Optional[str] = 'softmax') -> tf.Tensor:
    """Generates the output tensor of the Wide Residual Network.

    Args:
        classes: Number of output classes.
        img_input: Input tensor or layer.
        depth: Depth of the network. Compute N = (n - 4) / 6.
               For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
               For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
               For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
        widen_factor: Width of the network.
        dropout: Adds dropout if value is greater than 0.0.
        activation: activation function for the last dense layer.

    Returns:
        Output tensor of the network.
    """
    width = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    N = (depth - 4) // 6
    x = __conv1_block(img_input, width[0])

    for _ in range(N):
        x = __basic_block(x, width[1], dropout)

    x = layers.MaxPooling2D((2, 2))(x)

    for _ in range(N):
        x = __basic_block(x, width[2], dropout)

    x = layers.MaxPooling2D((2, 2))(x)

    for _ in range(N):
        x = __basic_block(x, width[3], dropout)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation=activation)(x)
    return x
