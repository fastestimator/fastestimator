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
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential, layers


# noinspection PyPep8Naming
def LeNet(input_shape: Tuple[int, int, int] = (28, 28, 1), classes: int = 10) -> tf.keras.Model:
    """A standard LeNet implementation in TensorFlow.

    The LeNet model has 3 convolution layers and 2 dense layers.

    Args:
        input_shape: shape of the input data (height, width, channels).
        classes: The number of outputs the model should generate.

    Raises:
        ValueError: Length of `input_shape` is not 3.
        ValueError: `input_shape`[0] or `input_shape`[1] is smaller than 18.

    Returns:
        A TensorFlow LeNet model.
    """
    _check_input_shape(input_shape)
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def _check_input_shape(input_shape):
    if len(input_shape) != 3:
        raise ValueError("Length of `input_shape` is not 3 (channel, height, width)")

    height, width, _ = input_shape

    if height < 18 or width < 18:
        raise ValueError("Both height and width of input_shape need to not smaller than 18")
