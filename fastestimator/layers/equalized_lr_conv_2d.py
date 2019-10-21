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
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers


class EqualizedLRConv2D(layers.Conv2D):
    """ Custom 2D convolutional layer used in PGGAN so that weights of each layer are updated at the same rate.
    """
    def __init__(self, filters, gain=np.sqrt(2), kernel_size=3, strides=(1, 1), padding="same"):
        super().__init__(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        self.gain = gain

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.float32(np.prod(self.kernel.shape[:-1]))
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def get_config(self):
        return {'filters': self.filters, 'gain': self.gain, "kernel_size": self.kernel_size}

    def call(self, x):
        return super().call(x) * self.wscale
