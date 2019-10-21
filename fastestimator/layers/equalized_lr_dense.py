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
import numpy as np

from tensorflow.keras import layers


class EqualizedLRDense(layers.Layer):
    """ Custom fully connected layer used in PGGAN so that weights of each layer are updated at the same rate.
    """
    def __init__(self, units, gain=np.sqrt(2)):
        super().__init__()
        self.units = units
        self.gain = gain

    def get_config(self):
        return {'units': self.units, 'gain': self.gain}

    def build(self, input_shape):
        self.w = self.add_weight(shape=[int(input_shape[-1]), self.units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
                                 trainable=True)
        fan_in = np.prod(input_shape[-1])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, x):
        return tf.matmul(x, self.w) * self.wscale
