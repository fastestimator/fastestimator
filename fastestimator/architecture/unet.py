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
"""U-Net architecture.
"""
from tensorflow.python.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model


def UNet(input_name, output_name, input_size=(128, 128, 3)):
    """U-Net model.

    Args:
        input_name (str): Input feature name.
        output_name (str): Output feature name.
        input_size (tuple, optional): Input image size. Defaults to (128, 128, 3).

    Returns:
        A `Model` object.
    """
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}

    inputs = Input(input_size, name=input_name)

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

    up6 = Conv2D(512, 2, **conv_config)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, **conv_config)(merge6)
    conv6 = Conv2D(512, 3, **conv_config)(conv6)

    up7 = Conv2D(256, 2, **conv_config)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, **conv_config)(merge7)
    conv7 = Conv2D(256, 3, **conv_config)(conv7)

    up8 = Conv2D(128, 2, **conv_config)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, **conv_config)(merge8)
    conv8 = Conv2D(128, 3, **conv_config)(conv8)

    up9 = Conv2D(64, 2, **conv_config)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, **conv_config)(merge9)
    conv9 = Conv2D(64, 3, **conv_config)(conv9)
    conv9 = Conv2D(2, 3, **conv_config)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', name=output_name)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
