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

@tf.custom_gradient
def reverse_grad(x, l):
    def custom_grad(dy):
        return -l*dy, None
    return tf.identity(x), custom_grad

class GradReversal(layers.Layer):

    def __init__(self, l=1.0):
        super().__init__()
        self.grl_const = l

    def call(self, x):
        return reverse_grad(x, self.grl_const)

    def get_config(self):
        return {"grl_const": self.grl_const}
