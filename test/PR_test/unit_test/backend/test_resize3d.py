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
        self.numpy_array = np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 3, 3, 3, 1))
        self.pytorch_array = to_tensor(np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 1, 3, 3, 3)), 'torch')
        self.tensorflow_array = to_tensor(np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 3, 3, 3, 1)), 'tf')

    def test_resize3d_compare(self):
        tensorflow_output = resize_3d(self.tensorflow_array, (4, 4, 4)).numpy()
        torch_output = resize_3d(self.pytorch_array, (4, 4, 4)).numpy()
        np.testing.assert_array_almost_equal(tensorflow_output[0, :, :, :, 0], torch_output[0, 0, :, :, :])

    def test_resize3d_torch(self):
        image_shape = resize_3d(self.pytorch_array, (4, 4, 4)).numpy().shape
        np.testing.assert_equal(image_shape, (1, 1, 4, 4, 4))

    def test_resize3d_tf(self):
        np.testing.assert_equal(resize_3d(self.tensorflow_array, (4, 4, 4)).numpy().shape, (1, 4, 4, 4, 1))
