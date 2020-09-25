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


class TestConcat(unittest.TestCase):
    def test_concat_np_input_axis_0(self):
        t = [np.array([[0, 1]]), np.array([[2, 3]]), np.array([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=0)
        obj2 = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_concat_np_input_axis_1(self):
        t = [np.array([[0, 1]]), np.array([[2, 3]]), np.array([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=1)
        obj2 = np.array([[0, 1, 2, 3, 4, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_concat_tf_input_axis_0(self):
        t = [tf.constant([[0, 1]]), tf.constant([[2, 3]]), tf.constant([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=0)
        obj2 = tf.constant([[0, 1], [2, 3], [4, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_concat_tf_input_axis_1(self):
        t = [tf.constant([[0, 1]]), tf.constant([[2, 3]]), tf.constant([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=1)
        obj2 = tf.constant([[0, 1, 2, 3, 4, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_concat_torch_input_axis_0(self):
        t = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=0)
        obj2 = torch.tensor([[0, 1], [2, 3], [4, 5]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_concat_torch_input_axis_1(self):
        t = [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])]
        obj1 = fe.backend.concat(t, axis=1)
        obj2 = torch.tensor([[0, 1, 2, 3, 4, 5]])
        self.assertTrue(is_equal(obj1, obj2))
