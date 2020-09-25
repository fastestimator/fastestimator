# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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
import unittest

import tensorflow as tf
import tensorflow_probability as tfp

import fastestimator as fe


class TestInstanceNorm(unittest.TestCase):
    def test_instance_norm(self):
        n = tfp.distributions.Normal(loc=10, scale=2)
        x = n.sample(sample_shape=(1, 100, 100, 1))
        m = fe.layers.tensorflow.InstanceNormalization()
        y = m(x)
        self.assertLess(tf.reduce_mean(y), 0.1)
        self.assertLess(tf.math.reduce_std(y), 0.1)
