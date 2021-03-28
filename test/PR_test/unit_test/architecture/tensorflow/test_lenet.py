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
from fastestimator.architecture.tensorflow.lenet import _check_input_shape


class TestLenet(unittest.TestCase):
    def test_lenet_default(self):
        data = np.ones((1, 28, 28, 1))
        input_data = tf.constant(data)
        lenet = LeNet()
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 10))

    def test_lenet_specific_input_size_classes(self):
        shape = (18, 18, 1)
        classes = 5
        data = np.ones((1, ) + shape)
        input_data = tf.constant(data)
        lenet = LeNet(input_shape=shape, classes=classes)
        output_shape = lenet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, classes))


class TestCheckInputShape(unittest.TestCase):
    def test_check_input_shape(self):
        with self.subTest("length not 3"):
            with self.assertRaises(ValueError):
                _check_input_shape((1, ))

        with self.subTest("width or height is smaller than 18"):
            with self.assertRaises(ValueError):
                _check_input_shape((13, 18, 1))

            with self.assertRaises(ValueError):
                _check_input_shape((1, 18, 1))

        with self.subTest("both are not smaller than 18"):
            _check_input_shape((18, 18, 1))
            _check_input_shape((18, 100, 32))
            _check_input_shape((200, 18, 64))
