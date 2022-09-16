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

from fastestimator.backend import to_tensor
from fastestimator.op.tensorop.normalize import Normalize


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.numpy_array = np.arange(0.0, 27.0, 1.0, dtype=np.float32).reshape((1, 3, 3, 3))
        self.numpy_array_int = np.arange(0, 27, 1, dtype=int).reshape((1, 3, 3, 3))
        self.expected_result = np.array(
            [[[[-1.6688062, -1.5404365, -1.4120668], [-1.283697, -1.1553273, -1.0269576], [
                -0.89858794, -0.77021825, -0.6418485
            ]], [[-0.5134788, -0.38510913, -0.2567394], [-0.1283697, 0., 0.1283697], [0.2567394, 0.38510913, 0.5134788]
                 ], [[0.6418485, 0.77021825, 0.89858794], [1.0269576, 1.1553273, 1.283697],
                     [1.4120668, 1.5404365, 1.6688062]]]],
            dtype=np.float32)
        self.expected_result_multi = np.array(
            [[[[-1.5331011, -1.543425, -1.5537487], [-1.1459544, -1.1562783, -1.166602],
               [-0.7588076, -0.7691315, -0.7794553]],
              [[-0.37166086, -0.38198477, -0.39230856], [0.01548585, 0.00516195, -0.00516183], [
                  0.4026326, 0.39230868, 0.3819849
              ]], [[0.7897793, 0.7794554, 0.7691316], [1.176926, 1.1666021, 1.1562784],
                   [1.5640727, 1.5537488, 1.5434251]]]],
            dtype=np.float32)
        self.numpy_array_torch = np.moveaxis(self.numpy_array, -1, 1)
        self.numpy_array_int_torch = np.moveaxis(self.numpy_array_int, -1, 1)
        self.expected_result_torch = np.moveaxis(self.expected_result, -1, 1)
        self.expected_result_multi_torch = np.moveaxis(self.expected_result_multi, -1, 1)


    def test_normalize_tf_int(self):
        op = Normalize(inputs="image", outputs="image", mean=0.482, std=0.289, max_pixel_value=27)
        data = op.forward(data=[tf.convert_to_tensor(self.numpy_array)], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result, 2)

    def test_normalize_tf_multi_int(self):
        op = Normalize(inputs="image",
                       outputs="image",
                       mean=(0.44, 0.48, 0.52),
                       std=(0.287, 0.287, 0.287),
                       max_pixel_value=27)
        data = op.forward(data=[tf.convert_to_tensor(self.numpy_array)], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result_multi, 2)

    def test_normalize_torch(self):
        op = Normalize(inputs="image", outputs="image", mean=0.482, std=0.289, max_pixel_value=27.0)
        data = op.forward(data=[to_tensor(self.numpy_array_int_torch, "torch")], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result_torch, 2)

    def test_normalize_torch_multi(self):
        op = Normalize(inputs="image",
                       outputs="image",
                       mean=(0.44, 0.48, 0.52),
                       std=(0.287, 0.287, 0.287),
                       max_pixel_value=27)
        data = op.forward(data=[to_tensor(self.numpy_array_int_torch, "torch")], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result_multi_torch, 2)

    def test_normalize_torch_float(self):
        op = Normalize(inputs="image", outputs="image", mean=0.482, std=0.289, max_pixel_value=27.0)
        data = op.forward(data=[to_tensor(self.numpy_array_torch, "torch")], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result_torch, 2)

    def test_normalize_torch_multi_float(self):
        op = Normalize(inputs="image",
                       outputs="image",
                       mean=(0.44, 0.48, 0.52),
                       std=(0.287, 0.287, 0.287),
                       max_pixel_value=27)
        data = op.forward(data=[to_tensor(self.numpy_array_torch, "torch")], state={})
        np.testing.assert_array_almost_equal(data[0].numpy(), self.expected_result_multi_torch, 2)

    def test_normalize_numpy_float(self):
        op = Normalize(inputs="image", outputs="image", mean=0.482, std=0.289, max_pixel_value=27.0)
        data = op.forward(data=[self.numpy_array], state={})
        np.testing.assert_array_almost_equal(data[0], self.expected_result, 2)

    def test_normalize_numpy_multi_float(self):
        op = Normalize(inputs="image",
                       outputs="image",
                       mean=(0.44, 0.48, 0.52),
                       std=(0.287, 0.287, 0.287),
                       max_pixel_value=27)
        data = op.forward(data=[self.numpy_array], state={})
        np.testing.assert_array_almost_equal(data[0], self.expected_result_multi, 2)
