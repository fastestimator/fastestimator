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
from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


def ResNet9(input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 10) -> tf.keras.Model:
    """A small 9-layer ResNet Tensorflow model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast

    Args:
        input_size: The size of the input tensor (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow ResNet9 model.
    """
    _check_input_size(input_size)

    # prep layers
    inp = layers.Input(shape=input_size)
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer1
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    # layer2
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer3
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    # layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(classes)(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    return model


def residual(x: KerasTensor, num_channel: int) -> KerasTensor:
    """A ResNet unit for ResNet9.

    Args:
        x: Input Keras tensor.
        num_channel: The number of layer channel.

    Return:
        Output Keras tensor.
    """
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def _check_input_size(input_size):
    if len(input_size) != 3:
        raise ValueError("Length of `input_size` is not 3 (channel, height, width)")

    height, width, _ = input_size

    if height < 16 or width < 16:
        raise ValueError("Both height and width of input_size need to not smaller than 16")
