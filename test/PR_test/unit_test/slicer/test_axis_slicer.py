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

from fastestimator.slicer import AxisSlicer


class TestAxisSlicer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = np.array([i for i in range(36)]).reshape((2, 3, 2, 3))

    def test_d0_slice(self):
        slicer = AxisSlicer(slice="x", axis=0)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 2)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[0])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 2)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[0])

    def test_d1_slice(self):
        slicer = AxisSlicer(slice="x", axis=1)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0, :, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0, :, :])

    def test_d2_slice(self):
        slicer = AxisSlicer(slice="x", axis=2)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 2)
            np.testing.assert_array_equal(minibatches[1].numpy(), self.batch[:, :, 1, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 2)
            np.testing.assert_array_equal(minibatches[1].numpy(), self.batch[:, :, 1, :])

    def test_dn1_slice(self):
        slicer = AxisSlicer(slice="x", axis=-1)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[1].numpy(), self.batch[:, :, :, 1])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[1].numpy(), self.batch[:, :, :, 1])

    def test_dn3_slice(self):
        slicer = AxisSlicer(slice="x", axis=-3)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[2].numpy(), self.batch[:, 2, :, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 3)
            np.testing.assert_array_equal(minibatches[2].numpy(), self.batch[:, 2, :, :])

    def test_d0_unslice(self):
        slicer = AxisSlicer(slice="x", unslice="x", axis=0)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)

    def test_d1_unslice(self):
        slicer = AxisSlicer(slice="x", unslice="x", axis=1)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)

    def test_d2_unslice(self):
        slicer = AxisSlicer(slice="x", unslice="x", axis=2)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)

    def test_dn1_unslice(self):
        slicer = AxisSlicer(slice="x", unslice="x", axis=-1)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)

    def test_dn3_unslice(self):
        slicer = AxisSlicer(slice="x", unslice="x", axis=-3)
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            combined = slicer._unslice_batch(tuple(minibatches), key='x')
            np.testing.assert_array_equal(combined, self.batch)
