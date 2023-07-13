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

import fastestimator as fe
from fastestimator.architecture.pytorch import UNet as UNet_Torch
from fastestimator.architecture.tensorflow import UNet as UNet_TF
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.slicer import MeanUnslicer, SlidingSlicer
from fastestimator.trace.metric import Dice


def _run_network_forward_slider_tf(q):
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

    q.put((estimator.system.global_step, float(result.history['eval']['Dice'][ds_size])))


class TestSlidingSlicer(unittest.TestCase):
    def test_network_forward_slider_tf(self):

        # We have to run TF tests in a sub-process to properly free GPU memory
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        runner = ctx.Process(target=_run_network_forward_slider_tf, args=(q, ))
        runner.start()
        runner.join()
        global_step, dice = q.get(block=False)
        q.close()

        self.assertEqual(global_step, 3)
        self.assertGreaterEqual(dice, 0)

    def test_network_forward_slider_torch(self):

        ds_size = 3

        sample_data = {
            "image": np.ones((ds_size, 1, 64, 64, 2), dtype=np.float32),
            "label": np.ones((ds_size, 6, 64, 64, 2), dtype=np.uint8)
        }

        dataset = NumpyDataset(data=sample_data)

        pipeline = fe.Pipeline(train_data=dataset, eval_data=dataset, batch_size=1, num_process=0)

        model = fe.build(
            model_fn=lambda: UNet_Torch(input_size=(1, 32, 32), output_channel=6),
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
                    window_size=(-1, -1, 32, 32, 1),
                    strides=(0, 0, 16, 16, 1),
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
