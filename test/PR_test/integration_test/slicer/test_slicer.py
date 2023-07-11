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
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import UNet as UNet_Torch
from fastestimator.architecture.tensorflow import UNet as UNet_TF
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.slicer import AxisSlicer, MeanUnslicer, Slicer, SlidingSlicer
from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.metric import Dice
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
    def test_network_transform_tf(self):

        sample_batch = {
            "image": torch.ones((1, 16, 16, 10, 1), dtype=torch.float32),
            "label": torch.zeros((1, 16, 16, 10, 6), dtype=torch.uint8)
        }

        model = fe.build(
            model_fn=lambda: UNet_TF(input_size=(16, 16, 1), output_channel=6),
            optimizer_fn=lambda: tf.optimizers.legacy.Adam(learning_rate=0.0001),
        )
        network = fe.Network(
            ops=[
                ModelOp(inputs="image", model=model, outputs="pred"),
                CrossEntropy(inputs=("pred", "label"), outputs="loss", form="binary"),
                UpdateOp(model=model, loss_name="loss")
            ],
            slicers=[AxisSlicer(slice=["image", "label"], unslice=["pred"], axis=3), MeanUnslicer(unslice="loss")])
        sample_prediction = network.transform(data=sample_batch, mode='train')
        self.assertEqual(sample_prediction['loss'].shape, [])
        np.testing.assert_array_almost_equal(sample_prediction["image"], sample_batch["image"])
        np.testing.assert_array_almost_equal(sample_prediction["label"], sample_batch["label"])
        self.assertEqual(sample_prediction["pred"].shape, sample_batch["label"].shape)

    def test_network_transform_torch(self):

        sample_batch = {
            "image": torch.ones((1, 1, 16, 16, 10), dtype=torch.float32),
            "label": torch.zeros((1, 6, 16, 16, 10), dtype=torch.uint8)
        }

        model = fe.build(
            model_fn=lambda: UNet_Torch(input_size=(1, 16, 16), output_channel=6),
            optimizer_fn="adam",
        )
        network = fe.Network(
            ops=[
                ModelOp(inputs="image", model=model, outputs="pred"),
                CrossEntropy(inputs=("pred", "label"), outputs="loss", form="binary"),
                UpdateOp(model=model, loss_name="loss")
            ],
            slicers=[AxisSlicer(slice=["image", "label"], unslice=["pred"], axis=-1), MeanUnslicer(unslice="loss")])
        sample_prediction = network.transform(data=sample_batch, mode='train')
        self.assertEqual(len(sample_prediction['loss'].shape), 0)
        np.testing.assert_array_almost_equal(sample_prediction["image"], sample_batch["image"])
        np.testing.assert_array_almost_equal(sample_prediction["label"], sample_batch["label"])
        self.assertEqual(sample_prediction["pred"].shape, sample_batch["label"].shape)

    def test_network_forward_torch(self):

        ds_size = 3

        sample_data = {
            "image": np.ones((ds_size, 1, 16, 16, 10), dtype=np.float32),
            "label": np.ones((ds_size, 6, 16, 16, 10), dtype=np.uint8)
        }

        dataset = NumpyDataset(data=sample_data)

        pipeline = fe.Pipeline(train_data=dataset, eval_data=dataset, batch_size=1)

        model = fe.build(
            model_fn=lambda: UNet_Torch(input_size=(1, 16, 16), output_channel=6),
            optimizer_fn="adam",
        )
        network = fe.Network(
            ops=[
                ModelOp(inputs="image", model=model, outputs="pred"),
                CrossEntropy(inputs=("pred", "label"), outputs="loss", form="binary"),
                UpdateOp(model=model, loss_name="loss")
            ],
            slicers=[AxisSlicer(slice=["image", "label"], unslice=["pred"], axis=-1), MeanUnslicer(unslice="loss")])

        estimator = fe.Estimator(pipeline=pipeline,
                                 network=network,
                                 traces=[Dice(true_key="label", pred_key="pred")],
                                 epochs=1)
        result = estimator.fit("test")

        self.assertEqual(estimator.system.global_step, ds_size)
        self.assertGreaterEqual(float(result.history['eval']['Dice'][ds_size]), 0)

    def test_network_forward_tf(self):

        ds_size = 3

        sample_data = {
            "image": np.ones((ds_size, 16, 16, 10, 1), dtype=np.float32),
            "label": np.ones((ds_size, 16, 16, 10, 6), dtype=np.uint8)
        }

        dataset = NumpyDataset(data=sample_data)

        pipeline = fe.Pipeline(train_data=dataset, eval_data=dataset, batch_size=1)

        model = fe.build(
            model_fn=lambda: UNet_TF(input_size=(16, 16, 1), output_channel=6),
            optimizer_fn=lambda: tf.optimizers.legacy.Adam(learning_rate=0.0001),
        )
        network = fe.Network(
            ops=[
                ModelOp(inputs="image", model=model, outputs="pred"),
                CrossEntropy(inputs=("pred", "label"), outputs="loss", form="binary"),
                UpdateOp(model=model, loss_name="loss")
            ],
            slicers=[AxisSlicer(slice=["image", "label"], unslice=["pred"], axis=3), MeanUnslicer(unslice="loss")])

        estimator = fe.Estimator(pipeline=pipeline,
                                 network=network,
                                 traces=[Dice(true_key="label", pred_key="pred")],
                                 epochs=1)
        result = estimator.fit("test")

        self.assertEqual(estimator.system.global_step, ds_size)
        self.assertGreaterEqual(float(result.history['eval']['Dice'][ds_size]), 0)

    def test_network_forward_slider_tf(self):

        ds_size = 3

        sample_data = {
            "image": np.ones((ds_size, 64, 64, 2, 1), dtype=np.float32),
            "label": np.ones((ds_size, 64, 64, 2, 6), dtype=np.uint8)
        }

        dataset = NumpyDataset(data=sample_data)

        pipeline = fe.Pipeline(train_data=dataset, eval_data=dataset, batch_size=1, num_process=0)

        model = fe.build(
            model_fn=lambda: UNet_TF(input_size=(32, 32, 1), output_channel=6),
            optimizer_fn="adam",
        )
        network = fe.Network(
            ops=[
                ModelOp(inputs="image", model=model, outputs="pred"),
                CrossEntropy(inputs=("pred", "label"), outputs="loss", form="binary"),
                UpdateOp(model=model, loss_name="loss")
            ],
            slicers=[
                SlidingSlicer(
                    slice=["image", "label"],
                    unslice=["pred"],
                    window_size=(-1, 32, 32, 1, -1),
                    strides=(0, 16, 16, 1, 0),
                    squeeze_window=True,
                ),
                MeanUnslicer(unslice="loss")
            ])

        estimator = fe.Estimator(pipeline=pipeline,
                                 network=network,
                                 traces=[Dice(true_key="label", pred_key="pred")],
                                 epochs=1)
        result = estimator.fit("test")

        self.assertEqual(estimator.system.global_step, ds_size)
        self.assertGreaterEqual(float(result.history['eval']['Dice'][ds_size]), 0)

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
