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

from fastestimator.backend import sparse_categorical_crossentropy


class TestSparseCategoricalCrossEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf_true = tf.constant([[0], [1], [0]])
        cls.tf_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
        cls.torch_true = torch.Tensor([[0], [1], [0]])
        cls.torch_pred = torch.Tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])

    def test_sparse_categorical_crossentropy_average_loss_true_tf(self):
        obj1 = sparse_categorical_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true).numpy()
        obj2 = 2.5336342
        self.assertTrue(np.allclose(obj1, obj2))

    def test_sparse_categorical_crossentropy_average_loss_false_tf(self):
        obj1 = sparse_categorical_crossentropy(y_pred=self.tf_pred, y_true=self.tf_true, average_loss=False).numpy()
        obj2 = np.array([2.3025851, 2.9957323, 2.3025851])
        self.assertTrue(np.allclose(obj1, obj2))

    def test_sparse_categorical_crossentropy_average_loss_true_torch(self):
        obj1 = sparse_categorical_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true).numpy()
        obj2 = 2.5336342
        self.assertTrue(np.allclose(obj1, obj2))

    def test_sparse_categorical_crossentropy_average_loss_false_torch(self):
        obj1 = sparse_categorical_crossentropy(y_pred=self.torch_pred, y_true=self.torch_true,
                                               average_loss=False).numpy()
        obj2 = np.array([2.3025851, 2.9957323, 2.3025851])
        self.assertTrue(np.allclose(obj1, obj2))
