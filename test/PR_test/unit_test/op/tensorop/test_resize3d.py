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
from fastestimator.op.tensorop.resize3d import Resize3D
from fastestimator.test.unittest_util import is_equal


class TestResize3d(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.input_data = np.reshape(range(1, 9), (1, 2, 2, 2, 1)).astype(np.float32)
        self.output_data = np.array([[[[[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]], [[3], [3], [4],
                                                                                                          [4]]],
                                      [[[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]], [[3], [3], [4],
                                                                                                          [4]]],
                                      [[[5], [5], [6], [6]], [[5], [5], [6], [6]], [[7], [7], [8], [8]], [[7], [7], [8],
                                                                                                          [8]]],
                                      [[[5], [5], [6], [6]], [[5], [5], [6], [6]], [[7], [7], [8], [8]],
                                       [[7], [7], [8], [8]]]]]).astype(np.float32)
        self.input_data_torch = np.moveaxis(self.input_data, -1, 1)
        self.output_data_torch = np.moveaxis(self.output_data, -1, 1)

    def test_output_shape(self):
        with self.assertRaises(ValueError):
            op = Resize3D(inputs="image", outputs="image", output_shape=[1, 8, 8, 8, 1])
            _ = op.forward(data=[to_tensor(self.input_data, 'tf')], state={})

    def test_resize_mode(self):
        with self.assertRaises(AssertionError):
            op = Resize3D(inputs="image", outputs="image", output_shape=[4, 4, 4], resize_mode="bilinear")
            _ = op.forward(data=[to_tensor(self.input_data, 'tf')], state={})

    def test_input_type(self):
        with self.assertRaises(ValueError):
            op = Resize3D(inputs="image", outputs="image", output_shape=[4, 4, 4], resize_mode="area")
            _ = op.forward(data=[self.input_data], state={})

    def test_resize_mode_nearest(self):
        op = Resize3D(inputs="image", outputs="image", output_shape=[4, 4, 4])
        data = op.forward(data=[to_tensor(self.input_data, 'tf')], state={})
        self.assertTrue(is_equal(data[0].numpy(), self.output_data))

        data = op.forward(data=[to_tensor(self.input_data_torch, 'torch')], state={})
        self.assertTrue(is_equal(data[0].numpy(), self.output_data_torch))

    def test_resize_mode_area(self):
        op = Resize3D(inputs="image", outputs="image", output_shape=[1, 1, 1], resize_mode='area')
        data = op.forward(data=[to_tensor(self.input_data, 'tf')], state={})
        self.assertEqual(data[0].numpy()[0, 0, 0, 0, 0], 4.5)

        data = op.forward(data=[to_tensor(self.input_data_torch, 'torch')], state={})
        self.assertEqual(data[0].numpy()[0, 0, 0, 0, 0], 4.5)
