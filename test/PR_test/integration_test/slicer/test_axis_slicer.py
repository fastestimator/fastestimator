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
import multiprocessing
import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import UNet as UNet_Torch
from fastestimator.architecture.tensorflow import UNet as UNet_TF
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.slicer import AxisSlicer, MeanUnslicer
from fastestimator.trace.metric import Dice


def _run_network_transform_tf(q):

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

    q.put((sample_prediction['loss'].shape,
           sample_prediction["pred"].shape,
           sample_prediction["image"].shape,
           sample_prediction["label"].shape))


def _run_network_forward_tf(q):
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
    q.put((estimator.system.global_step, float(result.history['eval']['Dice'][ds_size])))


class TestAxisSlicer(unittest.TestCase):
    def test_network_transform_tf(self):

        # We have to run TF tests in a sub-process to properly free GPU memory
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        runner = ctx.Process(target=_run_network_transform_tf, args=(q, ))
        runner.start()
        runner.join()
        loss_shape, pred_shape, image_shape, label_shape = q.get(block=False)
        q.close()

        self.assertEqual(loss_shape, [])
        self.assertEqual(image_shape, [1, 16, 16, 10, 1])
        self.assertEqual(pred_shape, [1, 16, 16, 10, 6])
        self.assertEqual(label_shape, [1, 16, 16, 10, 6])

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

        # We have to run TF tests in a sub-process to properly free GPU memory
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        runner = ctx.Process(target=_run_network_forward_tf, args=(q, ))
        runner.start()
        runner.join()
        global_step, dice = q.get(block=False)
        q.close()

        self.assertEqual(global_step, 3)
        self.assertGreaterEqual(dice, 0)
