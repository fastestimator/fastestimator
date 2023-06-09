# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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

from fastestimator.types import Array, DataSequence, Tensor


class TestTypes(unittest.TestCase):
    def test_tensor(self):
        with self.subTest('Torch Tensor'):
            self.assertTrue(isinstance(torch.ones((1)), Tensor))
        with self.subTest('TF Tensor'):
            self.assertTrue(isinstance(tf.ones((1)), Tensor))
        with self.subTest('TF Variable'):
            self.assertTrue(isinstance(tf.Variable(0.0), Tensor))
        with self.subTest('NP Array'):
            self.assertFalse(isinstance(np.ones((1)), Tensor))
        with self.subTest('List'):
            self.assertFalse(isinstance([1, 1, 0], Tensor))
        with self.subTest('Tuple'):
            self.assertFalse(isinstance((1, 1, 0), Tensor))
        with self.subTest('Dict'):
            self.assertFalse(isinstance({'a': 1}, Tensor))
        with self.subTest('Set'):
            self.assertFalse(isinstance({1, 2}, Tensor))

    def test_array(self):
        with self.subTest('Torch Tensor'):
            self.assertTrue(isinstance(torch.ones((1)), Array))
        with self.subTest('TF Tensor'):
            self.assertTrue(isinstance(tf.ones((1)), Array))
        with self.subTest('TF Variable'):
            self.assertTrue(isinstance(tf.Variable(0.0), Array))
        with self.subTest('NP Array'):
            self.assertTrue(isinstance(np.ones((1)), Array))
        with self.subTest('List'):
            self.assertFalse(isinstance([1, 1, 0], Array))
        with self.subTest('Tuple'):
            self.assertFalse(isinstance((1, 1, 0), Array))
        with self.subTest('Dict'):
            self.assertFalse(isinstance({'a': 1}, Array))
        with self.subTest('Set'):
            self.assertFalse(isinstance({1, 2}, Array))

    def test_data_sequence(self):
        with self.subTest('Torch Tensor'):
            self.assertTrue(isinstance(torch.ones((1)), DataSequence))
        with self.subTest('TF Tensor'):
            self.assertTrue(isinstance(tf.ones((1)), DataSequence))
        with self.subTest('TF Variable'):
            self.assertTrue(isinstance(tf.Variable(0.0), DataSequence))
        with self.subTest('NP Array'):
            self.assertTrue(isinstance(np.ones((1)), DataSequence))
        with self.subTest('List'):
            self.assertTrue(isinstance([1, 1, 0], DataSequence))
        with self.subTest('Tuple'):
            self.assertTrue(isinstance((1, 1, 0), DataSequence))
        with self.subTest('Dict'):
            self.assertFalse(isinstance({'a': 1}, DataSequence))
        with self.subTest('Set'):
            self.assertFalse(isinstance({1, 2}, DataSequence))
