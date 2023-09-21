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
import tempfile
import unittest
from typing import List, Tuple

import numpy as np

import fastestimator as fe
from fastestimator.op.tensorop import Delete
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.slicer import Slicer
from fastestimator.test.unittest_util import sample_system_object
from fastestimator.types import Array


class FakeSlicer(Slicer):
    def __init__(self, slice, unslice, mode, ds_id, var):
        super().__init__(slice=slice, unslice=unslice, mode=mode, ds_id=ds_id)
        self.var = var

    def _slice_batch(self, batch: Array) -> List[Array]:
        return [batch]

    def _unslice_batch(self, slices: Tuple[Array, ...], key: str) -> Array:
        return slices[0]


class TestSlicer(unittest.TestCase):
    def test_save_and_load_state_tf(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.tensorflow.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(
                ops=[ModelOp(model=model, inputs="x_out", outputs="y_pred")],
                slicers=FakeSlicer(slice="x", unslice=("x", "y_pred"), mode=None, ds_id=None, var=2.0))
            return system

        system = instantiate_system()

        # make some changes
        new_var = 4.0
        system.network.slicers[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.slicers[0].var

        self.assertEqual(loaded_var, new_var)

    def test_unslice_key_check(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(
                ops=[ModelOp(model=model, inputs="x", outputs="y_pred")],
                slicers=FakeSlicer(slice="x", unslice=("x", "y_pred"), mode=None, ds_id=None, var=2.0))
            return system

        system = instantiate_system()

        result = system.network.transform(data={'x': np.ones((1, 1, 28, 28), dtype=np.float32)}, mode="test")
        self.assertTupleEqual(tuple(result['x'].shape), (1,1,28,28))
        self.assertTupleEqual(tuple(result['y_pred'].shape), (1, 10))

    def test_unslice_key_check_missing_unslice(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(
                ops=[ModelOp(model=model, inputs="x", outputs="y_pred")],
                slicers=FakeSlicer(slice="x", unslice="x", mode=None, ds_id=None, var=2.0))
            return system

        system = instantiate_system()

        self.assertRaises(ValueError,
                          lambda: system.network.transform(data={'x': np.ones((1, 1, 28, 28), dtype=np.float32)},
                                                           mode="test"))

    def test_unslice_key_check_missing_deleted(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='tf')
            system.network = fe.Network(
                ops=[ModelOp(model=model, inputs="x", outputs="y_pred"),
                     Delete("y_pred")],
                slicers=FakeSlicer(slice="x", unslice="x", mode=None, ds_id=None, var=2.0))
            return system

        system = instantiate_system()

        result = system.network.transform(data={'x': np.ones((1, 1, 28, 28), dtype=np.float32)}, mode="test")
        self.assertTupleEqual(tuple(result['x'].shape), (1, 1, 28, 28))
        self.assertNotIn("y_pred", result)

    def test_save_and_load_state_torch(self):
        def instantiate_system():
            system = sample_system_object()
            model = fe.build(model_fn=fe.architecture.pytorch.LeNet, optimizer_fn='adam', model_name='torch')
            system.network = fe.Network(
                ops=[ModelOp(model=model, inputs="x_out", outputs="y_pred")],
                slicers=FakeSlicer(slice="x", unslice=("x", "y_pred"), mode=None, ds_id=None, var=2.0))
            return system

        system = instantiate_system()

        # make some changes
        new_var = 4.0
        system.network.slicers[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.network.slicers[0].var

        self.assertEqual(loaded_var, new_var)
