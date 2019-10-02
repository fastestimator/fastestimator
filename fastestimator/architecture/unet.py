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

def conv_block(input, nchannels, window, config, pooling=None, dropout=None):
    
    conv1 = Conv2D(nchannels, window, **config)(input)
    conv2 = Conv2D(nchannels, window, **config)(conv1)

    if dropout is not None:
        conv2 = Dropout(dropout)(conv2)

    if pooling is not None:
        pooled = MaxPooling2D(pool_size=(pooling, pooling))(conv2)
    else:
        pooled = None

    return conv2, pooled

def upsample(input, factor, nchannels, config):
    resized = UpSampling2D(size=(factor, factor))(input)
    up = Conv2D(nchannels, factor, **config)(resized) 
    
    return up

def up_concat(conv_pooled, conv, factor, nchannels, window, config, dropout=None):
    
    assert len(nchannels) == 2

    F, _ = conv_block(conv_pooled, nchannels[0], window, config, dropout=dropout)
    upsampled = upsample(F, factor, nchannels[1], config)
    feat = concatenate([conv, upsampled], axis=3)

    return feat

def UNet(input_size=(128, 128, 3), dropout=0.5, nblocks=5, nclasses=1):
    """Creates a U-Net model.
    This U-Net model is composed of nblocks "contracting blocks" and nblocks "expansive blocks".

    Args:
        input_size (tuple, optional): Shape of input image. Defaults to (128, 128, 3).
        dropout: If None, applies no dropout; Otherwise, applies dropout of probability equal 
                 to the parameter value (0-1 only)

    Returns:
        'Model' object: U-Net model.
    """
    
    assert dropout is None or 0 <= dropout <= 1, \
            "Invalid value for dropout parameter (None or 0 to 1 only)"  

    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}

    inputs = Input(input_size)
    
    C1, C1_pooled = conv_block(inputs, 64, 3, conv_config, pooling=2)
    C2, C2_pooled = conv_block(C1_pooled, 128, 3, conv_config, pooling=2)
    C3, C3_pooled = conv_block(C2_pooled, 256, 3, conv_config, pooling=2)
    C4, C4_pooled = conv_block(C3_pooled, 512, 3, conv_config, pooling=2, dropout=dropout)
    
    D4 = up_concat(C4_pooled, C4, 2, (1024, 512), 3, conv_config, dropout)
    D3 = up_concat(D4, C3, 2, (512, 256), 3, conv_config)
    D2 = up_concat(D3, C2, 2, (256, 128), 3, conv_config)
    D1 = up_concat(D2, C1, 2, (128, 64), 3, conv_config)
    
    C_end1, _ = conv_block(D1, 64, 3, conv_config)
    C_end2 = Conv2D(2, 3, **conv_config)(C_end1)

    y_dist = Conv2D(nclasses, 1, activation='sigmoid')(C_end2)

    model = Model(inputs=inputs, outputs=y_dist)

    return model
