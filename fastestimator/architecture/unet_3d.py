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
"""3D U-Net architecture."""

import tensorflow
from tensorflow.python.keras.layers import Activation, AveragePooling3D, BatchNormalization, Conv3D, Conv3DTranspose, \
    Input, MaxPooling3D, PReLU, SpatialDropout3D, UpSampling3D, concatenate
from tensorflow.python.keras.models import Model


def UNet3D(input_size=(9, 512, 512, 1),
           clip=(1024, 2048),
           dropout=0.5,
           nchannels=(32, 64, 128, 256),
           nclasses=1,
           bn=None,
           activation=lambda x: Activation('relu')(x),
           upsampling='copy',
           dilation_rates=(1, 1, 1, 1),
           residual=False):
    """Creates a U-Net model.
    This 3D U-Net model is composed of len(nchannels) "contracting blocks" and len(nchannels) "expansive blocks".

    Args:
        input_size (tuple, optional): Shape of input image. Defaults to (9, 512, 512, 1).
        clip: If not None, clips input values between clip[0] and clip[1]
        dropout: If None, applies no dropout; Otherwise, applies dropout of probability equal
                 to the parameter value (0-1 only)
        nchannels: Number of channels for each conv block; len(nchannels) decides number of blocks
        nclasses: Number of target classes for segmentation
        bn: [None, before, after] adds batchnorm layers across every convolution,
            before indicates adding BN before activation function is applied
            after indicates adding BN after activation function is applied
            Check https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md for related ablations!
        activation: Standard Keras activation functions
        upsampling: (copy, conv) Use copy for interpolation upsampling \
                    Use conv for transposed convolution based upsampling (learnable)
        dilation_rates: Add dilation to the encoder block conv layers [len(dilation_rates) == len(nchannels)]
        residual: False = no residual connections, True = residual connections in every layer

        NOTE: This particular model squashes down k 3D frames (batch * k * m * m * 1) into
              1 output frame (batch * 1 * m * m * nclasses).
              If different behavior is desired, please change the CNN model as necessary.

    Returns:
        'Model' object: U-Net model.
    """

    assert dropout is None or 0 <= dropout <= 1, "Invalid value for dropout parameter (None or 0 to 1 only)"

    assert bn in [None, "before", "after"], "Invalid bn parameter value"

    assert len(nchannels) >= 2, "At least 2 channels necessary for UNet"

    assert len(nchannels) == len(dilation_rates), "len(nchannels) should be the same as len(dilation_rates)"

    assert isinstance(residual, bool), 'Residual argument can be boolean only'

    # Handle callable activations as well
    if isinstance(activation, str):
        act = lambda x: Activation(activation)(x)
    else:
        act = activation

    inputs = Input(input_size)
    inp = inputs

    if clip:
        inputs = _clamp(inputs, clip[0], clip[1], min_max_scaling=True)

    levels = []

    # Contracting blocks
    for idx, nc in enumerate(nchannels):

        if idx == 0:
            W = (3, 3, 3)
        else:
            W = (1, 3, 3)

        C = _conv3D_block(inputs,
                          nc,
                          W,
                          nblocks=4,
                          dropout=dropout if dropout else 0,
                          bn="before",
                          prefix='3d_unet_conv_%d' % (idx),
                          activation=act,
                          dilation_rate=dilation_rates[idx],
                          residual=residual)

        if idx != len(nchannels) - 1:
            C_pooled = _pooling_combo3D(C, window=(1, 2, 2), prefix='3d_unet_pooling_%d' % (idx))
        else:
            C_pooled = None

        levels.append((C, C_pooled))

        if C_pooled is not None:
            inputs = C_pooled

    # Expanding blocks
    inp1, inp2 = levels[-1][0], levels[-2][0]
    for idx, nc in enumerate(reversed(nchannels[1:])):
        if idx == 0:
            d = dropout
            dilation = dilation_rates[-1]
        else:
            d = 0
            dilation = 1
        D = _up_concat(inp1,
                       inp2,
                       2, (nchannels[-1 - idx], nchannels[-2 - idx]),
                       3,
                       dropout=d,
                       bn=bn,
                       activation=act,
                       upsampling=upsampling,
                       dilation_rate=dilation,
                       idx=idx,
                       residual=residual)

        if idx != len(nchannels) - 2:
            inp1, inp2 = D, levels[-3 - idx][0]

    C_end1 = _conv3D_block(D, 64, (1, 3, 3), dropout=dropout, bn=bn, activation=act, residual=residual)

    C_end2 = Conv3D(2, (1, 3, 3), kernel_initializer='he_normal', padding='same')(C_end1)

    if bn:
        C_end2 = BatchNormalization()(C_end2)

    C_end2 = act(C_end2)

    y_dist = Conv3D(nclasses, 1, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(C_end2)

    model = Model(inputs=inp, outputs=y_dist)

    return model


def _conv3D_block(x,
                  nchannels,
                  window,
                  strides=(1, 1, 1),
                  nblocks=1,
                  dropout=0,
                  bn=None,
                  prefix='3d_unet_conv',
                  bias=False,
                  activation='relu',
                  dilation_rate=1,
                  residual=False):

    if isinstance(activation, str):
        act = Activation(activation)
    else:
        act = activation

    for i in range(nblocks):
        if residual:
            pad = 'same'
        else:
            pad = 'valid'

        inp = x
        x = Conv3D(filters=nchannels,
                   kernel_size=window,
                   strides=strides,
                   name=prefix + "_conv3d_" + str(i),
                   kernel_initializer='he_normal',
                   use_bias=bias,
                   dilation_rate=dilation_rate,
                   padding=pad)(x)

        if bn == 'before':
            x = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(i))(x)

        x = act(x)

        if bn == 'after':
            x = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(i))(x)

        if dropout > 0:
            x = SpatialDropout3D(rate=dropout, data_format='channels_last')(x)

        if inp.get_shape().as_list()[-1] == nchannels and residual:
            x = inp + x

    return x


def _deconv3D_block(x,
                    nchannels,
                    window,
                    strides=(1, 1, 1),
                    nblocks=1,
                    dropout=0,
                    prefix='unet_3d_deconv',
                    bias=False,
                    bn=None,
                    activation=Activation('relu'),
                    dilation_rate=1,
                    residual=False):

    if isinstance(activation, str):
        act = Activation(activation)
    else:
        act = activation

    for i in range(nblocks):
        if residual:
            pad = 'same'
        else:
            pad = 'valid'

        inp = x
        x = Conv3DTranspose(filters=nchannels,
                            kernel_size=window,
                            strides=strides,
                            name=prefix + "_deconv3d_" + str(i),
                            kernel_initializer='he_normal',
                            use_bias=bias,
                            dilation_rate=dilation_rate,
                            padding=pad)(x)

        if bn == 'before':
            x = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(i))(x)

        x = act(x)

        if bn == 'after':
            x = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(i))(x)

        if dropout > 0:
            x = SpatialDropout3D(rate=dropout, data_format='channels_last')(x)

        if inp.get_shape().as_list()[-1] == nchannels and residual:
            x = inp + x

    return x


def _pooling_combo3D(x, window=(2, 2, 2), prefix='3d_unet_pooling'):
    x = concatenate([
        MaxPooling3D(pool_size=window, data_format='channels_last', name=prefix + "_max")(x),
        AveragePooling3D(pool_size=window, data_format='channels_last', name=prefix + "_avg")(x)
    ],
                    axis=-1,
                    name=prefix + "_max_and_avg")

    return x


def _clamp(x, min_value, max_value, min_max_scaling=True):

    x = tensorflow.clip_by_value(x, clip_value_min=min_value, clip_value_max=max_value)

    if min_max_scaling:
        x = (x - min_value) / (max_value - min_value)

    return x


def _upsample(inp,
              factor,
              nchannels,
              bn=None,
              activation=None,
              bias=False,
              dilation_rate=1,
              prefix='unet_3d',
              idx=0,
              upsampling='copy',
              residual=False):

    if residual:
        resized = UpSampling3D(size=(1, factor, factor))(inp)
        resized = Conv3D(nchannels, (1, 1, 1), strides=1, padding='same')(resized)

        resized2 = Conv3DTranspose(nchannels, (1, factor, factor),
                                   strides=(1, factor, factor),
                                   name=prefix + "_deconv3d_" + str(idx),
                                   kernel_initializer='he_normal',
                                   use_bias=bias,
                                   dilation_rate=dilation_rate)(inp)
    else:
        if upsampling == 'copy':
            resized = UpSampling3D(size=(1, factor, factor))(inp)
            resized = Conv3D(nchannels, (1, 1, 1), strides=1, padding='same')(resized)
        else:
            resized = Conv3DTranspose(nchannels, (1, factor, factor),
                                      strides=(1, factor, factor),
                                      name=prefix + "_deconv3d_" + str(idx),
                                      kernel_initializer='he_normal',
                                      use_bias=bias,
                                      dilation_rate=dilation_rate)(inp)

    if bn == 'before':
        resized = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(idx))(resized)

    resized = activation(resized)

    if bn == 'after':
        resized = BatchNormalization(axis=4, name=prefix + "_batchnorm_" + str(idx))(resized)

    if inp.get_shape().as_list()[-1] == nchannels and residual:
        x = inp + x

    return resized


def _up_concat(conv_pooled,
               conv,
               factor,
               nchannels,
               window,
               strides=(1, 1, 1),
               dropout=None,
               bn=None,
               activation=None,
               idx=0,
               bias=False,
               dilation_rate=1,
               upsampling='copy',
               residual=False):

    assert len(nchannels) == 2

    F = _deconv3D_block(conv_pooled,
                        nchannels[0], (1, window, window),
                        strides=strides,
                        nblocks=4,
                        dropout=dropout,
                        prefix='unet_3d_deconv_%d' % (idx),
                        bias=bias,
                        bn=bn,
                        activation=activation,
                        residual=residual)

    upsampled = _upsample(F,
                          factor,
                          nchannels[1],
                          bn=bn,
                          activation=activation,
                          upsampling=upsampling,
                          bias=bias,
                          dilation_rate=dilation_rate,
                          prefix='unet_3d',
                          idx=idx,
                          residual=residual)

    feat = concatenate([conv, upsampled], axis=-1)

    return feat
