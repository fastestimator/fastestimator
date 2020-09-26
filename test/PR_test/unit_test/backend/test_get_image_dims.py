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
import torch

from fastestimator.backend import get_image_dims
from fastestimator.test.unittest_util import is_equal


class TestGetImageDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_np = np.random.random((2, 12, 12, 3))
        cls.test_tf = tf.random.uniform((2, 12, 12, 3))
        cls.test_torch = torch.rand((2, 3, 12, 12))
        cls.test_output = (3, 12, 12)

    def test_np_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_np), self.test_output))

    def test_tf_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_tf), self.test_output))

    def test_torch_value(self):
        self.assertTrue(is_equal(get_image_dims(self.test_torch), self.test_output))
