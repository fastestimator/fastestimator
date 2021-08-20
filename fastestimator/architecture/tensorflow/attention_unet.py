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
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, ReLU, \
    UpSampling2D, concatenate
from tensorflow.python.keras.models import Model


# noinspection PyPep8Naming
def AttentionUNet(input_size: Tuple[int, int, int] = (128, 128, 1)) -> tf.keras.Model:
    """Attention based UNet implementation in TensorFlow.

    Args:
        input_size: The size of the input tensor (height, width, channels).

    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.

    Returns:
        A TensorFlow Attention UNet model.
    """
    _check_input_size(input_size)
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    up_config = {'size': (2, 2), 'interpolation': 'bilinear'}
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, **conv_config)(inputs)
    conv1 = Conv2D(64, 3, **conv_config)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, **conv_config)(pool1)
    conv2 = Conv2D(128, 3, **conv_config)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, **conv_config)(pool2)
    conv3 = Conv2D(256, 3, **conv_config)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, **conv_config)(pool3)
    conv4 = Conv2D(512, 3, **conv_config)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, **conv_config)(pool4)
    conv5 = Conv2D(1024, 3, **conv_config)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 3, **conv_config)(UpSampling2D(**up_config)(drop5))
    drop4 = attention_block(512, decoder_input=up6, encoder_input=drop4)
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, **conv_config)(merge6)
    conv6 = Conv2D(512, 3, **conv_config)(conv6)

    up7 = Conv2D(256, 3, **conv_config)(UpSampling2D(**up_config)(conv6))
    conv3 = attention_block(256, decoder_input=up7, encoder_input=conv3)
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(256, 3, **conv_config)(merge7)
    conv7 = Conv2D(256, 3, **conv_config)(conv7)

    up8 = Conv2D(128, 3, **conv_config)(UpSampling2D(**up_config)(conv7))
    conv2 = attention_block(128, decoder_input=up8, encoder_input=conv2)
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(128, 3, **conv_config)(merge8)
    conv8 = Conv2D(128, 3, **conv_config)(conv8)

    up9 = Conv2D(64, 3, **conv_config)(UpSampling2D(**up_config)(conv8))
    conv1 = attention_block(64, decoder_input=up9, encoder_input=conv1)
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(64, 3, **conv_config)(merge9)
    conv9 = Conv2D(64, 3, **conv_config)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model


def attention_block(n_filters: int, decoder_input: tf.Tensor, encoder_input: tf.Tensor) -> tf.Tensor:
    """An attention unit for Attention Unet.

    Args:
        n_filters: How many filters for the convolution layer.
        decoder_input: Input tensor in the decoder section.
        encoder_input: Input tensor in the encoder section.

    Return:
        Output Keras tensor.
    """
    c1 = Conv2D(n_filters, kernel_size=1)(decoder_input)
    c1 = BatchNormalization()(c1)
    x1 = Conv2D(n_filters, kernel_size=1)(encoder_input)
    x1 = BatchNormalization()(x1)
    att = ReLU()(x1 + c1)
    att = Conv2D(1, kernel_size=1)(att)
    att = BatchNormalization()(att)
    att = tf.sigmoid(att)
    return encoder_input * att


def _check_input_size(input_size):
    if len(input_size) != 3:
        raise ValueError("Length of `input_size` is not 3 (channel, height, width)")

    height, width, _ = input_size

    if height < 16 or not (height / 16.0).is_integer() or width < 16 or not (width / 16.0).is_integer():
        raise ValueError("Both height and width of input_size need to be multiples of 16 (16, 32, 48...)")
