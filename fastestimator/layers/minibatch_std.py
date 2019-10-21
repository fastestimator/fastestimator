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
from tensorflow.keras import layers


class MiniBatchStd(layers.Layer):
    """ A layer that outputs a concatenation of the input tensor and an average of channel-wise standard deviation of the input tensor

    Args:
        group_size (int): a parameter determining size of subgroup to compute the statistics
    """
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def get_config(self):
        return {'group_size': self.group_size}

    def call(self, x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        s = x.shape  # [NHWC]
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC]
        y = tf.reduce_mean(tf.square(y), axis=0)  #[MHWC]
        y = tf.sqrt(y + 1e-8)  # [MHWC]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.tile(y, [group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([x, y], axis=-1)
