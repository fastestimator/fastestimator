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
from typing import List, Tuple

import tensorflow as tf

from fastestimator.slicer import Slicer
from fastestimator.slicer.slicer import forward_slicers, reverse_slicers, sanity_assert_slicers
from fastestimator.types import Array


class MockSlicer(Slicer):
    def _slice_batch(self, batch: Array) -> List[Array]:
        return [batch]

    def _unslice_batch(self, slices: Tuple[Array, ...], key: str) -> Array:
        return slices[0]


class MockSlicer2(Slicer):
    def _slice_batch(self, batch: Array) -> List[Array]:
        return [batch, batch]

    def _unslice_batch(self, slices: Tuple[Array, ...], key: str) -> Array:
        return slices[0]


class MockSlicerSliceOnly(Slicer):
    def _slice_batch(self, batch: Array) -> List[Array]:
        return [batch]


class MockSlicerUnSliceOnly(Slicer):
    def _unslice_batch(self, slices: Tuple[Array, ...], key: str) -> Array:
        return slices[0]


class TestSlicer(unittest.TestCase):
    def test_raise_from_init(self):
        with self.subTest("No inputs"):
            self.assertRaises(ValueError, lambda: MockSlicer(slice=()))
        with self.subTest("Undefined Slice"):
            self.assertRaises(NotImplementedError, lambda: MockSlicerUnSliceOnly(slice="x"))
        with self.subTest("Undefined UnSlice"):
            self.assertRaises(NotImplementedError, lambda: MockSlicerSliceOnly(slice="y", unslice="x"))


class TestSanityAssert(unittest.TestCase):
    def test_sanity(self):
        with self.subTest("Mix slice keys"):
            slicers = [MockSlicer(slice=("x", "y")), MockSlicer(slice="z"), MockSlicer(slice=("y", "b"))]
            self.assertRaises(ValueError, lambda: sanity_assert_slicers(slicers))
        with self.subTest("Mix unslice keys"):
            slicers = [
                MockSlicer(slice=("x", "y")),
                MockSlicer(slice="z", unslice="z"),
                MockSlicer(slice=("a", "b"), unslice="z")
            ]
            self.assertRaises(ValueError, lambda: sanity_assert_slicers(slicers))


class TestForwardSlicers(unittest.TestCase):
    def test_forward(self):
        with self.subTest("Single Slice"):
            slicers = [MockSlicer(slice="x"), MockSlicer(slice=("y", "z"))]
            batch = {"x": tf.ones((3, 5, 5, 7)), "y": tf.ones((3, 10)), "z": tf.ones((3, 1)), "w": tf.ones((3, 2))}
            minibatch = forward_slicers(slicers=slicers, data=batch)
            self.assertEqual(len(minibatch), 1)
            self.assertDictEqual(minibatch[0], batch)
        with self.subTest("Multi Slice"):
            slicers = [MockSlicer2(slice="x"), MockSlicer2(slice=("y", "z"))]
            batch = {"x": tf.ones((3, 5, 5, 7)), "y": tf.ones((3, 10)), "z": tf.ones((3, 1)), "w": tf.ones((3, 2))}
            minibatch = forward_slicers(slicers=slicers, data=batch)
            self.assertEqual(len(minibatch), 2)
            self.assertDictEqual(minibatch[0], batch)


class TestReverseSlicers(unittest.TestCase):
    def test_reverse(self):
        slicers = [MockSlicer(slice="x"), MockSlicer(slice=("y", "z"))]
        batch = {"x": tf.ones((3, 5, 5, 7)), "y": tf.ones((3, 10)), "z": tf.ones((3, 1)), "w": tf.ones((3, 2))}
        minibatch = forward_slicers(slicers=slicers, data=batch)
        for elem in minibatch:
            elem.pop("z")  # simulate z being un-wanted later
        final_batch = reverse_slicers(slicers=slicers, data=minibatch, original_data={})
        batch.pop("z")
        self.assertDictEqual(final_batch, batch)
