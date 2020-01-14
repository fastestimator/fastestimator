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

from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv3D, Input, LeakyReLU, ReLU, \
    SpatialDropout3D, UpSampling3D, concatenate
from tensorflow.python.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def convolution_module(layer,
                       filters,
                       kernel_size=(3, 3, 3),
                       strides=(1, 1, 1),
                       padding='same',
                       instance_norm=False,
                       batch_norm=False,
                       activation=None):
    layer = Conv3D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   data_format='channels_first')(layer)
    if instance_norm == True:
        layer = InstanceNormalization(axis=1)(layer)
    if batch_norm == True:
        layer = BatchNormalization(axis=1)(layer)
    if activation is None:
        layer = ReLU()(layer)
    else:
        layer = activation()(layer)
    return layer


def context_module(layer, filters, dropout):
    layer = convolution_module(layer, filters, activation=LeakyReLU, instance_norm=True)
    layer = SpatialDropout3D(rate=dropout, data_format='channels_first')(layer)
    layer = convolution_module(layer, filters, activation=LeakyReLU, instance_norm=True)
    return layer


def upsampling_module(layer, filters):
    upsample = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(layer)
    upsample_conv = convolution_module(upsample, filters, activation=LeakyReLU, instance_norm=True)
    return upsample_conv


def localization_module(layer, filters):
    layer = convolution_module(layer, filters, activation=LeakyReLU, instance_norm=True)
    layer = convolution_module(layer, filters, activation=LeakyReLU, kernel_size=(1, 1, 1), instance_norm=True)
    return layer


def UNet3D_Isensee(input_shape=(4, 128, 128, 128),
                   nchannels=(16, 32, 64, 128, 256),
                   n_labels=3,
                   activation='sigmoid',
                   n_segmentation_level=3):
    inputs = Input(input_shape)
    current_layer = inputs
    downsample_level = []
    segmentation_level = []
    for idx, filters in enumerate(nchannels):
        if idx == 0:
            in_conv = convolution_module(current_layer, filters, activation=LeakyReLU, instance_norm=True)
        else:
            in_conv = convolution_module(current_layer,
                                         filters,
                                         activation=LeakyReLU,
                                         strides=(2, 2, 2),
                                         instance_norm=True)
        context_layer = context_module(in_conv, filters, dropout=0.3)
        add_layer = Add()([in_conv, context_layer])
        downsample_level.append(add_layer)
        current_layer = add_layer

    for idx, filters in reversed(list(enumerate(nchannels[:-1:]))):
        upsample_conv = upsampling_module(current_layer, filters)
        current_layer = concatenate([upsample_conv, downsample_level[idx]], axis=1)
        localization = localization_module(current_layer, filters)
        current_layer = localization
        if idx < n_segmentation_level:
            layer = Conv3D(n_labels, kernel_size=(1, 1, 1), data_format='channels_first')(localization)
            segmentation_level.append(layer)

    output_layer = None
    for idx in range(len(segmentation_level)):
        if output_layer is None:
            output_layer = segmentation_level[idx]
        else:
            output_layer = Add()([output_layer, segmentation_level[idx]])
        if idx != len(segmentation_level) - 1:
            output_layer = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(output_layer)

    act = Activation(activation)(output_layer)
    model = Model(inputs=inputs, outputs=act)
    return model
