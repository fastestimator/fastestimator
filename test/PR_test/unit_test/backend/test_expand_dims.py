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

import fastestimator as fe
from fastestimator.test.unittest_util import is_equal


class TestExpandDims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np_input = np.array([2, 7, 5])
        cls.torch_input = torch.tensor([2, 7, 5])
        cls.tf_input = tf.constant([2, 7, 5])

    def test_expand_dims_np_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.np_input, axis=0)
        obj2 = np.array([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_np_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.np_input, axis=1)
        obj2 = np.array([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_tf_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.tf_input, axis=0)
        obj2 = tf.constant([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_tf_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.tf_input, axis=1)
        obj2 = tf.constant([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_torch_input_axis_0(self):
        obj1 = fe.backend.expand_dims(self.torch_input, axis=0)
        obj2 = torch.tensor([[2, 7, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_expand_dims_torch_input_axis_1(self):
        obj1 = fe.backend.expand_dims(self.torch_input, axis=1)
        obj2 = torch.tensor([[2], [7], [5]])
        self.assertTrue(is_equal(obj1, obj2))
