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


class TestGatherFromBatch(unittest.TestCase):
    def test_gather_from_batch_np_input(self):
        tensor = np.array([[0, 1], [2, 3], [4, 5]])
        indice = np.array([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = np.array([1, 2, 5])
        self.assertTrue(is_equal(obj1, obj2))

    def test_gather_from_batch_tf_input(self):
        tensor = tf.constant([[0, 1], [2, 3], [4, 5]])
        indice = tf.constant([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = tf.constant([1, 2, 5])
        self.assertTrue(is_equal(obj1, obj2))

        tensor = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = tf.constant([[2, 3], [4, 5], [10, 11]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_gather_from_batch_torch_input_2D(self):
        tensor = torch.tensor([[0, 1], [2, 3], [4, 5]])
        indice = torch.tensor([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = torch.tensor([1, 2, 5])
        self.assertTrue(is_equal(obj1, obj2))

    def test_gather_from_batch_torch_input_3D(self):
        tensor = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        indice = torch.tensor([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = torch.tensor([[2, 3], [4, 5], [10, 11]])
        self.assertTrue(is_equal(obj1, obj2))

    def test_gather_from_batch_input_and_indice_diff_type_2D(self):
        tensor = torch.tensor([[0, 1], [2, 3], [4, 5]])
        indice = np.array([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = torch.tensor([1, 2, 5])
        self.assertTrue(is_equal(obj1, obj2))

    def test_gather_from_batch_input_and_indice_diff_type_3D(self):
        tensor = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]])
        indice = torch.tensor([1, 0, 1])
        obj1 = fe.backend.gather_from_batch(tensor, indice)
        obj2 = tf.constant([[2, 3], [4, 5], [10, 11]])
        self.assertTrue(is_equal(obj1, obj2))
