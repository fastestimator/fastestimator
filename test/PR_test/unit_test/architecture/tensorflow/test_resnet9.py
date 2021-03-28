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

from fastestimator.architecture.tensorflow import ResNet9
from fastestimator.architecture.tensorflow.resnet9 import _check_input_size


class TestResNet9(unittest.TestCase):
    def test_resnet9_default(self):
        data = np.ones((1, 32, 32, 3))
        input_data = tf.constant(data)
        model = ResNet9()
        output_shape = model(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 10))

    def test_resnet9_specific_input_size_classes(self):
        size = (16, 16, 1)
        classes = 5
        data = np.ones((1, ) + size)
        input_data = tf.constant(data)
        model = ResNet9(input_size=size, classes=classes)
        output_shape = model(input_data).numpy().shape
        self.assertEqual(output_shape, (1, classes))


class TestCheckInputSize(unittest.TestCase):
    def test_resnet9_check_input_size(self):
        with self.subTest("length not 3"):
            with self.assertRaises(ValueError):
                _check_input_size((1, ))

        with self.subTest("width or height is smaller than 16"):
            with self.assertRaises(ValueError):
                _check_input_size((13, 16, 1))

            with self.assertRaises(ValueError):
                _check_input_size((16, 1, 1))

        with self.subTest("both are not smaller than 16"):
            _check_input_size((16, 16, 1))
            _check_input_size((16, 100, 3))
