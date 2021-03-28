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
import torch

from fastestimator.architecture.pytorch import LeNet


class TestLenet(unittest.TestCase):
    def test_lenet_default(self):
        data = np.ones((1, 1, 28, 28))
        input_data = torch.Tensor(data)
        lenet = LeNet()
        output_shape = lenet(input_data).detach().numpy().shape
        self.assertEqual(output_shape, (1, 10))

    def test_lenet_specific_input_shape_classes(self):
        size = (1, 18, 18)
        classes = 3
        data = np.ones((1, ) + size)
        input_data = torch.Tensor(data)
        lenet = LeNet(size, classes=classes)
        output_shape = lenet(input_data).detach().numpy().shape
        self.assertEqual(output_shape, (1, classes))

    def test_check_input_shape(self):
        with self.subTest("length not 3"):
            with self.assertRaises(ValueError):
                LeNet._check_input_shape((1, ))

        with self.subTest("width or height is smaller than 18"):
            with self.assertRaises(ValueError):
                LeNet._check_input_shape((1, 13, 18))

            with self.assertRaises(ValueError):
                LeNet._check_input_shape((1, 18, 1))

        with self.subTest("both are not smaller than 18"):
            LeNet._check_input_shape((1, 18, 18))
            LeNet._check_input_shape((32, 18, 100))
            LeNet._check_input_shape((64, 200, 18))
