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

from fastestimator.architecture.tensorflow import AttentionUNet
from fastestimator.architecture.tensorflow.attention_unet import _check_input_size


class TestAttentionUNet(unittest.TestCase):
    def test_attention_unet_default(self):
        data = np.ones((1, 128, 128, 1))
        input_data = tf.constant(data)
        attention_unet = AttentionUNet()
        output_shape = attention_unet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, 128, 128, 1))

    def test_attention_unet_specific_input_size(self):
        size = (16, 16, 1)
        data = np.ones((1, ) + size)
        input_data = tf.constant(data)
        attention_unet = AttentionUNet(input_size=size)
        output_shape = attention_unet(input_data).numpy().shape
        self.assertEqual(output_shape, (1, ) + size)


class TestCheckInputSize(unittest.TestCase):
    def test_check_input_size(self):
        with self.subTest("length not 3"):
            with self.assertRaises(ValueError):
                _check_input_size((1, ))

        with self.subTest("width or height is not a multiple of 16"):
            with self.assertRaises(ValueError):
                _check_input_size((18, 16, 1))

            with self.assertRaises(ValueError):
                _check_input_size((32, 100, 1))

            with self.assertRaises(ValueError):
                _check_input_size((48, 0, 1))

        with self.subTest("both are multiples of 16"):
            _check_input_size((16, 48, 1))
            _check_input_size((128, 64, 3))
