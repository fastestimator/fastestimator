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


class ApplyBias(layers.Layer):
    """Class to be used along with EqualizedLRDense and EqualizedLRConv2D for PGGAN

    """
    def build(self, input_shape):
        self.b = self.add_weight(shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, x):
        return x + self.b
