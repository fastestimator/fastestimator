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
from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.python.keras import layers


class InstanceNormalization(layers.Layer):
    """A layer for performing instance normalization.

    This class is intentionally not @traceable (models and layers are handled by a different process).

    This layer assumes that you are using the a tensor shaped like (Batch, Height, Width, Channels). See
    https://arxiv.org/abs/1607.08022 for details about this layer. The implementation here is borrowed from
    https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py.

    ```python
    n = tfp.distributions.Normal(loc=10, scale=2)
    x = n.sample(sample_shape=(1, 100, 100, 1))  # mean ~= 10, stddev ~= 2
    m = fe.layers.tensorflow.InstanceNormalization()
    y = m(x)  # mean ~= 0, stddev ~= 0
    ```

    Args:
        epsilon: A numerical stability constant added to the variance.
    """
    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None

    def get_config(self) -> Dict[str, Any]:
        return {'epsilon': self.epsilon}

    def build(self, input_shape: Tuple[int, int, int, int]) -> None:
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1:],
                                     initializer=tf.random_normal_initializer(0., 0.02),
                                     trainable=True)

        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
