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
"""U-Net architecture."""

from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input, \
    MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model


def UNet(input_size=(128, 128, 3),
         dropout=0.5,
         nchannels=(64, 128, 256, 512, 1024),
         nclasses=1,
         bn=None,
         activation='relu',
         upsampling='bilinear',
         dilation_rates=(1, 1, 1, 1, 1)):
    """Creates a U-Net model.
    This U-Net model is composed of len(nchannels) "contracting blocks" and len(nchannels) "expansive blocks".

    Args:
        input_size (tuple, optional): Shape of input image. Defaults to (128, 128, 3).
        dropout: If None, applies no dropout; Otherwise, applies dropout of probability equal
                 to the parameter value (0-1 only)
        nchannels: Number of channels for each conv block; len(nchannels) decides number of blocks
        nclasses: Number of target classes for segmentation
        bn: [None, before, after] adds batchnorm layers across every convolution,
            before indicates adding BN before activation function is applied
            after indicates adding BN after activation function is applied
            Check https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md for related ablations!
        activation: Standard Keras activation functions
        upsampling: (bilinear, nearest, conv) Use bilinear, nearest (nearest neighbour) for predetermined upsampling \
                    Use conv for transposed convolution based upsampling (learnable)
        dilation_rates: Add dilation to the encoder block conv layers [len(dilation_rates) == len(nchannels)]

    Returns:
        'Model' object: U-Net model.
    """

    assert dropout is None or 0 <= dropout <= 1, "Invalid value for dropout parameter (None or 0 to 1 only)"

    assert bn in [None, "before", "after"], "Invalid bn parameter value"

    assert len(nchannels) >= 2, "At least 2 channels necessary for UNet"

    assert len(nchannels) == len(dilation_rates), "len(nchannels) should be the same as len(dilation_rates)"

    # Handle callable activations as well
    if isinstance(activation, str):
        act = activation
    else:
        act = None

    conv_config = {'activation': act, 'padding': 'same', 'kernel_initializer': 'he_normal'}

    inputs = Input(input_size)
    inp = inputs

    levels = []

    # Contracting blocks
    for idx, nc in enumerate(nchannels[:-1]):

        if idx == len(nchannels) - 2:
            d = dropout
        else:
            d = None

        C, C_pooled = conv_block(inputs, nc, 3, conv_config, pooling=2, dropout=d, bn=bn, activation=activation, dilation_rate=dilation_rates[idx])
        levels.append((C, C_pooled))
        inputs = C_pooled

    # Expanding blocks
    inp1, inp2 = levels[-1][1], levels[-1][0]
    for idx, nc in enumerate(reversed(nchannels[1:])):
        if idx == 0:
            d = dropout
            dilation = dilation_rates[-1]
        else:
            d = None
            dilation = None

        D = up_concat(inp1,
                      inp2,
                      2, (nchannels[-1 - idx], nchannels[-2 - idx]),
                      3,
                      conv_config,
                      d,
                      bn=bn,
                      activation=activation,
                      upsampling=upsampling,
                      dilation=dilation)
        if idx != len(nchannels) - 2:
            inp1, inp2 = D, levels[-2 - idx][0]

    C_end1, _ = conv_block(D, 64, 3, conv_config, bn=bn, activation=activation)

    if bn:
        if bn == 'before':
            act = conv_config['activation']
            conv_config['activation'] = None

    C_end2 = Conv2D(2, 3, **conv_config)(C_end1)

    if bn:
        C_end2 = BatchNormalization()(C_end2)

        if bn == 'before':
            if act:
                C_end2 = Activation(act)(C_end2)
            else:
                C_end2 = activation(C_end2)

    y_dist = Conv2D(nclasses, 1, activation='sigmoid')(C_end2)

    model = Model(inputs=inp, outputs=y_dist)

    return model


def conv_block(inp, nchannels, window, config, pooling=None, dropout=None, bn=False, activation=None, dilation_rate=1):

    if bn and bn == 'before':
        act = config['activation']
        config['activation'] = None

    conv1 = Conv2D(nchannels, window, dilation_rate=dilation_rate, **config)(inp)

    if bn:
        conv1 = BatchNormalization()(conv1)

        if bn == 'before':
            if act:
                conv1 = Activation(act)(conv1)
            else:
                conv1 = activation(conv1)

    conv2 = Conv2D(nchannels, window, dilation_rate=dilation_rate, **config)(conv1)

    if bn:
        conv2 = BatchNormalization()(conv2)

        if bn == 'before':
            if act:
                conv2 = Activation(act)(conv2)
            else:
                conv2 = activation(conv2)

            config['activation'] = act  # python dicts are reference based

    if dropout:
        conv2 = Dropout(dropout)(conv2)

    if pooling:
        pooled = MaxPooling2D(pool_size=(pooling, pooling))(conv2)
    else:
        pooled = None

    return conv2, pooled


def upsample(inp, factor, nchannels, config, bn=None, activation=None, upsampling='bilinear'):

    if upsampling in ['bilinear', 'nearest']:
        resized = UpSampling2D(size=(factor, factor), interpolation=upsampling)(inp)
    else:
        resized = Conv2DTranspose(nchannels, factor, strides=(factor, factor), padding='same')(inp)

    if bn and bn == 'before':
        act = config['activation']
        config['activation'] = None

    up = Conv2D(nchannels, factor, **config)(resized)

    if bn:
        up = BatchNormalization()(up)

        if bn == 'before':
            if act:
                up = Activation(act)(up)
            else:
                up = activation(up)

            config['activation'] = act

    return up


def up_concat(conv_pooled,
              conv,
              factor,
              nchannels,
              window,
              config,
              dropout=None,
              bn=None,
              activation=None,
              upsampling='bilinear',
              dilation=1):

    assert len(nchannels) == 2

    F, _ = conv_block(conv_pooled, nchannels[0], window, config, bn=bn, dropout=dropout, activation=activation, dilation_rate=dilation if dilation else 1)
    upsampled = upsample(F, factor, nchannels[1], config, bn=bn, activation=activation, upsampling=upsampling)
    feat = concatenate([conv, upsampled], axis=3)

    return feat
