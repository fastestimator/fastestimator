# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.backend import to_tensor, normalize


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.numpy_array = np.arange(0.0, 12.0, 1.0, dtype=np.float32).reshape((1, 2, 2, 3))
        self.numpy_array_int = np.arange(0.0, 12.0, 1.0, dtype=int).reshape((1, 2, 2, 3))
        self.expected_result = np.array([[[[-1.593255, -1.3035723, -1.0138896], [-0.7242068, -0.4345241, -0.14484136]],
                                          [[0.14484136, 0.4345241, 0.7242068], [1.0138896, 1.3035723, 1.593255]]]],
                                        dtype=np.float32)

        self.numpy_array_float = np.moveaxis(self.numpy_array, -1, 1)
        self.numpy_array_int_float = np.moveaxis(self.numpy_array_int, -1, 1)
        self.expected_result_torch = np.moveaxis(self.expected_result, -1, 1)

    def test_normalize_np_value(self):
        np.testing.assert_array_almost_equal(normalize(self.numpy_array, 0.5, 0.31382295, 11.0), self.expected_result)

    def test_normalize_np_value_int(self):
        np.testing.assert_array_almost_equal(normalize(self.numpy_array_int, 0.5, 0.31382295, 11), self.expected_result)

    def test_normalize_tf_value(self):
        np.testing.assert_array_almost_equal(
            normalize(tf.convert_to_tensor(self.numpy_array), 0.5, 0.31382295, 11.0).numpy(), self.expected_result)

    def test_normalize_tf_value_int(self):
        np.testing.assert_array_almost_equal(
            normalize(tf.convert_to_tensor(self.numpy_array_int), 0.5, 0.31382295, 11.0).numpy(), self.expected_result)

    def test_normalize_torch_value(self):
        np.testing.assert_array_almost_equal(
            normalize(to_tensor(self.numpy_array_float, 'torch'), 0.5, 0.31382295, 11.0).numpy(), self.expected_result_torch)

    def test_normalize_torch_value_int(self):
        np.testing.assert_array_almost_equal(
            normalize(to_tensor(self.numpy_array_int_float, 'torch'), 0.5, 0.31382295, 11.0).numpy(),
            self.expected_result_torch)
