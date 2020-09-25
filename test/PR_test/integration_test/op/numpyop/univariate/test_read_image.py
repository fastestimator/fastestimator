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
import os
import unittest

import numpy as np

from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.test.unittest_util import is_equal


class TestReadImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1_path = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "..", "util", "resources", "test_read_image.png"))
        cls.img2_path = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "..", "util", "resources", "test_read_image2.png"))
        cls.expected_image_output = np.zeros((28, 28, 3))
        cls.expected_second_image_output = 255 * np.ones((28, 28, 3))

    def test_single_input(self):
        data = [self.img1_path]
        image = ReadImage(inputs='x', outputs='x')
        output = image.forward(data=data, state={})
        self.assertTrue(is_equal(output[0], self.expected_image_output))

    def test_multi_input(self):
        data = [self.img1_path, self.img2_path]
        image = ReadImage(inputs='x', outputs='x')
        output = image.forward(data=data, state={})
        with self.subTest('Check first image in data'):
            self.assertTrue(is_equal(output[0], self.expected_image_output))
        with self.subTest('Check second image in data'):
            self.assertTrue(is_equal(output[1], self.expected_second_image_output))
