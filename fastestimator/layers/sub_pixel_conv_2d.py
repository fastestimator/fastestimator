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

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras import Model, layers


class SubPixelConv2D(layers.Layer):
    """ Class for upsampling using subpixel convolution (https://arxiv.org/pdf/1609.05158.pdf)

        Args:
            upsample_factor (int, optional): [description]. Defaults to 2.
            nchannels (int, optional): [description]. Defaults to 128.
    """
    def __init__(self, upsample_factor=2, nchannels=128):

        assert isinstance(upsample_factor, int) and upsample_factor > 1, 'Invalid upsample_factor'

        self.factor = upsample_factor
        self.nchannels = nchannels
        super().__init__()

    def get_config(self):
        """Get JSON config for params

        Returns:
            [dict]: params defining subpixel convolution layer
        """
        return {'upsample_factor': self.factor, 'nchannels': self.nchannels}

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return tuple([s[0], s[1] * self.factor, s[2] * self.factor, self.nchannels])

    def build(self, input_shape):

        self.shape_conv = Conv2D(self.factor * self.factor * self.nchannels, 1, 1, padding='same')
        super().build(input_shape)

    def call(self, x):

        b, h, w, c = x.get_shape().as_list()

        if c // (self.factor * self.factor) != self.nchannels:
            x = self.shape_conv(x)

        y = tf.nn.depth_to_space(x, block_size=self.factor)

        return y
