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

from fastestimator.slicer import SlidingSlicer
from fastestimator.slicer.slicer import forward_slicers, reverse_slicers


class TestSlidingSlicer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = np.array([i for i in range(600)], dtype=np.int16).reshape((2, 10, 10, 3))
        cls.padded_batch = np.zeros(shape=(2, 13, 13, 3), dtype=cls.batch.dtype)
        cls.padded_batch[:, :10, :10, :] += cls.batch
        cls.padded_batch[:, :, 10:, :] -= 1
        cls.padded_batch[:, 10:, :, :] -= 1

    def test_noop_slice(self):
        slicer = SlidingSlicer(slice="x", window_size=(-1, 10, -1, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 1)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 1)
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch)

    def test_gap_stride_pad_nopad(self):
        slicer = SlidingSlicer(slice="x", pad_mode='nopad', window_size=(-1, 2, 4, 3), strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 5:9, :])

        slicer = SlidingSlicer(slice="x", pad_mode='nopad', window_size=(-1, 3, 2, 4), strides=(0, 0, 3, 5))
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            padded_batch = torch.moveaxis(torch.Tensor(self.padded_batch), -1, 1)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 3, 2, 4])
            np.testing.assert_array_equal(minibatches[0].numpy(), batch[:, :, 0:2, 0:4])
            np.testing.assert_array_equal(minibatches[-1].numpy(), padded_batch[:, :, 8:10, 5:9])

    def test_tiled_slice(self):
        slicer = SlidingSlicer(slice="x", window_size=(-1, 2, 2, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])

    def test_tiled_unequal_slice(self):
        slicer = SlidingSlicer(slice="x", window_size=(-1, 2, 5, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 10)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 5, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:5, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 10)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 5, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:5, :])

    def test_drop_unequal_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='drop', window_size=(-1, 2, 4, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 10)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 10)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])

    def test_partial_tiled_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='partial', window_size=(-1, 2, 2, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])

    def test_partial_unequal_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='partial', window_size=(-1, 2, 4, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for idx, mbatch in enumerate(minibatches):
                if idx % 3 == 2:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
                else:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for idx, mbatch in enumerate(minibatches):
                if idx % 3 == 2:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
                else:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])

    def test_pad_tiled_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='constant', pad_val=-1, window_size=(-1, 2, 2, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 8:10, 8:10, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 8:10, 8:10, :])

    def test_pad_tiled_slice_mirror(self):
        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 2, 2, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 2, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:2, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 8:10, 8:10, :])

        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 3, 2, 2))
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 25)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 3, 2, 2])
            np.testing.assert_array_equal(minibatches[0].numpy(), batch[:, :, 0:2, 0:2])
            np.testing.assert_array_equal(minibatches[-1].numpy(), batch[:, :, 8:10, 8:10])

    def test_pad_unequal_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='constant', pad_val=-1, window_size=(-1, 2, 4, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 8:12, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 8:12, :])

    def test_pad_unequal_slice_mirror(self):
        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 2, 4, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy()[:, :, 3, :], self.padded_batch[:, 8:10, 7, :])

        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 3, 2, 4))
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            padded_batch = torch.moveaxis(torch.Tensor(self.padded_batch), -1, 1)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 3, 2, 4])
            np.testing.assert_array_equal(minibatches[0].numpy(), batch[:, :, 0:2, 0:4])
            np.testing.assert_array_equal(minibatches[-1].numpy()[:, :, :, 3], padded_batch[:, :, 8:10, 7])

    def test_overlapping_stride_pad(self):
        slicer = SlidingSlicer(slice="x",
                               pad_mode='constant',
                               pad_val=-1,
                               window_size=(-1, 2, 4, 3),
                               strides=(0, 2, 3, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 6:10, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 6:10, :])

    def test_overlapping_stride_pad_mirror(self):
        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 2, 4, 3), strides=(0, 2, 3, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 8:10, 6:10, :])

        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 3, 2, 4), strides=(0, 0, 2, 3))
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            padded_batch = torch.moveaxis(torch.Tensor(self.padded_batch), -1, 1)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 3, 2, 4])
            np.testing.assert_array_equal(minibatches[0].numpy(), batch[:, :, 0:2, 0:4])
            np.testing.assert_array_equal(minibatches[-1].numpy(), padded_batch[:, :, 8:10, 6:10])

    def test_overlapping_stride_drop(self):
        slicer = SlidingSlicer(slice="x", pad_mode='drop', pad_val=-1, window_size=(-1, 2, 4, 3), strides=(0, 2, 3, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 8:10, 6:10, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 8:10, 6:10, :])

    def test_overlapping_partial_unequal_slice(self):
        slicer = SlidingSlicer(slice="x", pad_mode='partial', window_size=(-1, 2, 4, 3), strides=(0, 2, 3, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for idx, mbatch in enumerate(minibatches):
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 15)
            for idx, mbatch in enumerate(minibatches):
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])

    def test_gap_stride_pad(self):
        slicer = SlidingSlicer(slice="x",
                               pad_mode='constant',
                               pad_val=-1,
                               window_size=(-1, 2, 4, 3),
                               strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 9:11, 5:9, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.padded_batch[:, 9:11, 5:9, :])

    def test_gap_stride_pad_mirror(self):
        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 2, 4, 3), strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy()[:, 1, :, :], self.padded_batch[:, 8, 5:9, :])

        slicer = SlidingSlicer(slice="x", pad_mode='mirror', window_size=(-1, 3, 2, 4), strides=(0, 0, 3, 5))
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            padded_batch = torch.moveaxis(torch.Tensor(self.padded_batch), -1, 1)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 3, 2, 4])
            np.testing.assert_array_equal(minibatches[0].numpy(), batch[:, :, 0:2, 0:4])
            np.testing.assert_array_equal(minibatches[-1].numpy()[:, :, 1, :], padded_batch[:, :, 8, 5:9])

    def test_gap_stride_drop(self):
        slicer = SlidingSlicer(slice="x", pad_mode='drop', window_size=(-1, 2, 4, 3), strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 6)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 6:8, 5:9, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 6)
            for mbatch in minibatches:
                self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
            np.testing.assert_array_equal(minibatches[-1].numpy(), self.batch[:, 6:8, 5:9, :])

    def test_gap_stride_partial(self):
        slicer = SlidingSlicer(slice="x", pad_mode='partial', window_size=(-1, 2, 4, 3), strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for idx, mbatch in enumerate(minibatches):
                if idx in (6, 7):
                    self.assertListEqual(list(mbatch.shape), [2, 1, 4, 3])
                else:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = slicer._slice_batch(batch)
            self.assertEqual(len(minibatches), 8)
            for idx, mbatch in enumerate(minibatches):
                if idx in (6, 7):
                    self.assertListEqual(list(mbatch.shape), [2, 1, 4, 3])
                else:
                    self.assertListEqual(list(mbatch.shape), [2, 2, 4, 3])
            np.testing.assert_array_equal(minibatches[0].numpy(), self.batch[:, 0:2, 0:4, :])

    def test_tiled_unslice(self):
        slicer = SlidingSlicer(slice="x", window_size=(-1, 2, 2, 3))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 25)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 25)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)

    def test_overlapping_drop_avg_unslice(self):
        slicer = SlidingSlicer(slice="x",
                               window_size=(-1, 2, 4, -1),
                               strides=(0, 2, 3, 0),
                               pad_mode='drop',
                               pad_val=-1.0,
                               unslice_mode="avg")
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)

    def test_overlapping_partial_avg_unslice(self):
        slicer = SlidingSlicer(slice="x",
                               window_size=(-1, 2, 4, -1),
                               strides=(0, 2, 3, 0),
                               pad_mode='partial',
                               pad_val=-1.0,
                               unslice_mode="avg")
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)

    def test_overlapping_pad_avg_unslice(self):
        slicer = SlidingSlicer(slice="x",
                               window_size=(-1, 2, 4, -1),
                               strides=(0, 2, 3, 0),
                               pad_mode='constant',
                               pad_val=-1.0,
                               unslice_mode="avg")
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)


    def test_overlapping_pad_avg_unslice_mirror(self):
        slicer = SlidingSlicer(slice="x",
                               window_size=(-1, 2, 4, -1),
                               strides=(0, 2, 3, 0),
                               pad_mode='mirror',
                               unslice_mode="avg")
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), self.batch)

        slicer = SlidingSlicer(slice="x",
                               window_size=(-1, -1, 2, 4),
                               strides=(0, 0, 2, 3),
                               pad_mode='mirror',
                               unslice_mode="avg")
        with self.subTest("Torch"):
            batch = torch.moveaxis(torch.Tensor(self.batch), -1, 1)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 15)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 3, 10, 10])
            np.testing.assert_array_equal(combined.numpy(), batch)

    def test_gap_stride_partial_unslice(self):
        pad_val = -2
        gap_batch = self.batch.copy()
        gap_batch[:, :, 4::5, :] = pad_val
        gap_batch[:, 2::3, :, :] = pad_val
        slicer = SlidingSlicer(slice="x",
                               pad_mode='partial',
                               pad_val=pad_val,
                               window_size=(-1, 2, 4, 3),
                               strides=(0, 3, 5, 0))
        with self.subTest("TF"):
            batch = tf.convert_to_tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 8)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), gap_batch)
        with self.subTest("Torch"):
            batch = torch.Tensor(self.batch)
            minibatches = forward_slicers([slicer], data={'x': batch})
            self.assertEqual(len(minibatches), 8)
            combined = reverse_slicers([slicer], minibatches, original_data={})
            combined = combined['x']
            self.assertListEqual(list(combined.shape), [2, 10, 10, 3])
            np.testing.assert_array_equal(combined.numpy(), gap_batch)
