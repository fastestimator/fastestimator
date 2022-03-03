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

from fastestimator.backend import to_tensor
from fastestimator.backend.resize3d import resize_3d


class TestResize3D(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pytorch_array = to_tensor(np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 1, 3, 3, 3)), 'torch')
        self.tensorflow_array = to_tensor(np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 3, 3, 3, 1)), 'tf')
        self.expected_result = (1, 4, 4, 4)

    '''def test_normalize_np_value(self):
        np.testing.assert_array_almost_equal(normalize(self.numpy_array, 0.5, 0.31382295, 11.0), self.expected_result)

    def test_normalize_np_value_int(self):
        np.testing.assert_array_almost_equal(normalize(self.numpy_array_int, 0.5, 0.31382295, 11), self.expected_result)

    def test_normalize_tf_value(self):
        np.testing.assert_array_almost_equal(
            normalize(tf.convert_to_tensor(self.numpy_array), 0.5, 0.31382295, 11.0).numpy(), self.expected_result)

    def test_normalize_tf_value_int(self):
        np.testing.assert_array_almost_equal(
            normalize(tf.convert_to_tensor(self.numpy_array_int), 0.5, 0.31382295, 11.0).numpy(), self.expected_result)'''

    def test_resize3d_tf(self):
        np.testing.assert_equal(resize_3d(self.tensorflow_array, (4, 4, 4)).numpy().shape, self.expected_result)

    def test_resize3d_torch(self):
        image_shape = resize_3d(self.pytorch_array, (4, 4, 4)).numpy().shape
        print(image_shape)
        np.testing.assert_equal(image_shape, self.expected_result)
