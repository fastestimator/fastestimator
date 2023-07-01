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

from fastestimator.slicer import MeanUnslicer


class TestMeanUnslicer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minibatches = [np.array([0.3, 0.6, 0.9]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])]
        cls.target = np.array([1.1, 1.2, 1.3])

    def test_unslice(self):
        slicer = MeanUnslicer(unslice="x")
        with self.subTest("TF"):
            minibatches = [tf.convert_to_tensor(elem) for elem in self.minibatches]
            batch = slicer._unslice_batch(minibatches, key='x')
            np.testing.assert_array_almost_equal(batch, self.target)
        with self.subTest("Torch"):
            minibatches = [torch.tensor(elem) for elem in self.minibatches]
            batch = slicer._unslice_batch(minibatches, key='x')
            np.testing.assert_array_almost_equal(batch, self.target)
