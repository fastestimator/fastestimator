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
from tensorflow.python.keras import Model, layers


# Code borrowed from https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(layers.Layer):
    """Class for performing Reflection Padding on 2D arrays.

    Args:
        padding (tuple, optional): padding size. Defaults to (1, 1).
    """
    def __init__(self, padding=(1, 1)):
        self.padding = tuple(padding)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super().__init__()

    def get_config(self):
        return {'padding': self.padding}

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')
