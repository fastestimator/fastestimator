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


class TestTensorPow(unittest.TestCase):
    def test_np_input_pow_gt_1(self):
        n = np.array([[1, 4, 6], [2.3, 0.5, 0]])
        target = np.array([[1, 8.44485063e+01, 3.09089322e+02], [1.43723927e+01, 1.08818820e-01, 0]])
        b = fe.backend.tensor_pow(n, 3.2)
        self.assertTrue(np.allclose(b, target))

    def test_np_input_pow_lt_1(self):
        n = np.array([[1, 4, 6], [2.3, 0.5, 0]])
        target = np.array([[1, 1.33792755, 1.45683968], [1.1911401, 0.86453723, 0]])
        b = fe.backend.tensor_pow(n, 0.21)
        self.assertTrue(np.allclose(b, target))

    def test_tf_input_pow_gt_1(self):
        n = tf.convert_to_tensor([[1, 4, 6], [2.3, 0.5, 0]])
        target = tf.convert_to_tensor([[1, 8.44485063e+01, 3.09089322e+02], [1.43723927e+01, 1.08818820e-01, 0]])
        b = fe.backend.tensor_pow(n, 3.2)
        self.assertTrue(np.allclose(b, target))

    def test_tf_input_pow_lt_1(self):
        n = tf.convert_to_tensor([[1, 4, 6], [2.3, 0.5, 0]])
        target = tf.convert_to_tensor([[1, 1.33792755, 1.45683968], [1.1911401, 0.86453723, 0]])
        b = fe.backend.tensor_pow(n, 0.21)
        self.assertTrue(np.allclose(b, target))

    def test_torch_input_pow_gt_1(self):
        n = torch.tensor([[1, 4, 6], [2.3, 0.5, 0]])
        target = torch.tensor([[1, 8.44485063e+01, 3.09089322e+02], [1.43723927e+01, 1.08818820e-01, 0]])
        b = fe.backend.tensor_pow(n, 3.2)
        self.assertTrue(np.allclose(b, target))

    def test_torch_input_pow_lt_1(self):
        n = torch.tensor([[1, 4, 6], [2.3, 0.5, 0]])
        target = torch.tensor([[1, 1.33792755, 1.45683968], [1.1911401, 0.86453723, 0]])
        b = fe.backend.tensor_pow(n, 0.21)
        self.assertTrue(np.allclose(b, target))
