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

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestReflectionPadding2D(unittest.TestCase):
    def setUp(self):
        self.x = tf.reshape(tf.convert_to_tensor(list(range(9))), (1, 3, 3, 1))

    def test_reflection_padding_2d_double_side(self):
        op = tf.constant([[[[4], [3], [4], [5], [4]], [[1], [0], [1], [2], [1]], [[4], [3], [4], [5], [4]],
                           [[7], [6], [7], [8], [7]], [[4], [3], [4], [5], [4]]]])
        m = fe.layers.tensorflow.ReflectionPadding2D((1, 1))
        y = m(self.x)
        self.assertTrue(fet.is_equal(y, op))

    def test_reflection_padding_2d_single_side(self):
        op = tf.constant([[[[1], [0], [1], [2], [1]], [[4], [3], [4], [5], [4]], [[7], [6], [7], [8], [7]]]])
        m = fe.layers.tensorflow.ReflectionPadding2D((1, 0))
        y = m(self.x)
        self.assertTrue(fet.is_equal(y, op))
