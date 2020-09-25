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

import numpy as np
import tensorflow as tf

from fastestimator.architecture.tensorflow import LeNet


class TestLenet(unittest.TestCase):
    def test_lenet(self):
        data = np.ones((1, 28, 28, 1))
        input_data = tf.constant(data)
        lenet = LeNet()
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 10))

    def test_lenet_class(self):
        data = np.ones((1, 28, 28, 1))
        input_data = tf.constant(data)
        lenet = LeNet(classes=5)
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 5))
