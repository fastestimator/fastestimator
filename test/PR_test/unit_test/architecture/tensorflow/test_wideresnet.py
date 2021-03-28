# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.architecture.tensorflow import WideResidualNetwork


class TestWideResNet(unittest.TestCase):
    def test_wrn(self):
        data = np.ones((1, 32, 32, 3))
        input_data = tf.constant(data)
        wrn = WideResidualNetwork(input_shape=(32, 32, 3), classes=5)
        output_shape = wrn(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 5))

    def test_wrn_depth(self):
        with self.assertRaises(ValueError):
            WideResidualNetwork(input_shape=(32, 32, 3), depth=27)
