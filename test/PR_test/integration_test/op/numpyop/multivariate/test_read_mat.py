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

from fastestimator.op.numpyop.multivariate import ReadMat
from fastestimator.test.unittest_util import is_equal


class TestReadMat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat_path = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "..", "util", "resources", "test_read_mat.mat"))
        cls.expected_mat_output = np.arange(20).reshape(1, 20)
        cls.expected_second_image_output = 255 * np.ones((28, 28, 3))

    def test_input_multiple_keys(self):
        image = ReadMat(file='x', keys=['a', 'label'])
        output = image.forward(data=self.mat_path, state={})
        with self.subTest('Check data in mat'):
            self.assertTrue(is_equal(output[0], self.expected_mat_output))
        with self.subTest('Check label in mat'):
            self.assertTrue(is_equal(output[1], np.array(['testcase'])))
