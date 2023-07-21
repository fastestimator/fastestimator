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
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

from fastestimator.util.data import Data

if TYPE_CHECKING:
    from tensorflow.python.keras import Sequential, layers
else:
    from tensorflow.keras import Sequential, layers

import fastestimator as fe
from fastestimator.dataset.interleave_dataset import InterleaveDataset
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.numpyop import Batch, NumpyOp
from fastestimator.op.tensorop.model.model import ModelOp
from fastestimator.op.tensorop.tensorop import LambdaOp
from fastestimator.test.unittest_util import sample_system_object, sample_system_object_torch
from fastestimator.trace.trace import Trace
from fastestimator.util import get_num_gpus


class Plus(NumpyOp):
    def forward(self, data, state):
        return data + 0.5


class TestDataset(NumpyDataset):
    def __init__(self, data, var):
        super().__init__(data)
        self.var = var


class TestInterleaveDatasetRestoreWizard(unittest.TestCase):
    def test_save_and_load_state_with_batch_dataset_tf(self):
        def instantiate_system():
            system = sample_system_object()
            x_train = np.ones((2, 28, 28, 3))
            y_train = np.ones((2, ))
            ds = TestDataset(data={'x': x_train, 'y': y_train}, var=1)
            train_data = InterleaveDataset(datasets=[ds, ds])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=2)
            return system

        system = instantiate_system()

        # make some change
        new_var = 2
        system.pipeline.data["train"][''].datasets[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # reinstantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.data["train"][''].datasets[0].var

        self.assertEqual(loaded_var, new_var)

    def test_save_and_load_state_with_batch_dataset_torch(self):
        def instantiate_system():
            system = sample_system_object_torch()
            x_train = np.ones((2, 3, 28, 28))
            y_train = np.ones((2, ))
            ds = TestDataset(data={'x': x_train, 'y': y_train}, var=1)
            train_data = InterleaveDataset(datasets=[ds, ds])
            system.pipeline = fe.Pipeline(train_data=train_data, batch_size=2)
            return system

        system = instantiate_system()

        # make some change
        new_var = 2
        system.pipeline.data["train"][''].datasets[0].var = new_var

        # save the state
        save_path = tempfile.mkdtemp()
        system.save_state(save_path)

        # re-instantiate system and load the state
        system = instantiate_system()
        system.load_state(save_path)
        loaded_var = system.pipeline.data["train"][''].datasets[0].var

        self.assertEqual(loaded_var, new_var)


class TestInterleaveDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data1 = {"x": [x for x in range(1, 11)], "ds_id": [0 for _ in range(10)]}
        self.data2 = {"x": [x * 10 for x in range(1, 21)], "ds_id": [1 for _ in range(20)]}

    def test_interleave_in_pipeline_default_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets=[ds1, ds2])
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2)
        with pipeline(mode="train", shuffle=True) as loader:
            batches = [batch for batch in loader]
        self.assertAlmostEqual(batches[0]['ds_id'].numpy().sum(), 0)
        self.assertAlmostEqual(batches[1]['ds_id'].numpy().sum(), 2)
        self.assertAlmostEqual(batches[2]['ds_id'].numpy().sum(), 0)
        self.assertAlmostEqual(batches[3]['ds_id'].numpy().sum(), 2)

    def test_interleave_in_pipeline_custom_pattern(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets=[ds1, ds2], pattern=[0, 0, 1])
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2)
        with pipeline(mode="train", shuffle=True) as loader:
            batches = [batch for batch in loader]
        self.assertAlmostEqual(batches[0]['ds_id'].numpy().sum(), 0)
        self.assertAlmostEqual(batches[1]['ds_id'].numpy().sum(), 0)
        self.assertAlmostEqual(batches[2]['ds_id'].numpy().sum(), 2)
        self.assertAlmostEqual(batches[3]['ds_id'].numpy().sum(), 0)

    def test_interleave_in_pipeline_ds_tags(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2, ops=[Plus(inputs="x", outputs="x", ds_id="ds1")])
        results = pipeline.get_results(mode="train", num_steps=4)
        self.assertAlmostEqual(results[0]['x'].numpy()[0] % 1, 0.5)
        self.assertAlmostEqual(results[1]['x'].numpy()[0] % 1, 0)
        self.assertAlmostEqual(results[2]['x'].numpy()[0] % 1, 0.5)
        self.assertAlmostEqual(results[3]['x'].numpy()[0] % 1, 0)

    def test_interleave_in_pipeline_ds_tags_with_multids_id(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        pipeline = fe.Pipeline(
            train_data={"source1": dataset},
            batch_size=2,
            ops=[Plus(inputs="x", outputs="x", ds_id="ds1"), Plus(inputs="x", outputs="x", ds_id="source1")])
        results = pipeline.get_results(mode="train", num_steps=4, ds_id="source1")
        self.assertAlmostEqual(results[0]['x'].numpy()[0] % 1, 0)
        self.assertAlmostEqual(results[1]['x'].numpy()[0] % 1, 0.5)
        self.assertAlmostEqual(results[2]['x'].numpy()[0] % 1, 0)
        self.assertAlmostEqual(results[3]['x'].numpy()[0] % 1, 0.5)

    def test_interleave_in_pipeline_with_different_batch_size_hybrid(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2, ops=[Batch(batch_size=1, ds_id="ds2")])
        results = pipeline.get_results(mode="train", num_steps=4)
        self.assertAlmostEqual(len(results[0]['x'].numpy()), 2)
        self.assertAlmostEqual(len(results[1]['x'].numpy()), 1)
        self.assertAlmostEqual(len(results[2]['x'].numpy()), 2)
        self.assertAlmostEqual(len(results[3]['x'].numpy()), 1)

    def test_interleave_in_pipeline_with_different_batch_size_op(self):
        ds1, ds2 = NumpyDataset(self.data1), NumpyDataset(self.data2)
        dataset = InterleaveDataset(datasets={"ds1": ds1, "ds2": ds2})
        pipeline = fe.Pipeline(train_data=dataset,
                               ops=[Batch(batch_size=2, ds_id="ds1"), Batch(batch_size=1, ds_id="ds2")])
        results = pipeline.get_results(mode="train", num_steps=4)
        self.assertAlmostEqual(len(results[0]['x'].numpy()), 2)
        self.assertAlmostEqual(len(results[1]['x'].numpy()), 1)
        self.assertAlmostEqual(len(results[2]['x'].numpy()), 2)
        self.assertAlmostEqual(len(results[3]['x'].numpy()), 1)

    def test_interleave_dataset_with_different_shapes(self):
        def mymodel(input_shape):
            model = Sequential()
            model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=input_shape))
            return model

        class Collector(Trace):
            def __init__(self) -> None:
                super().__init__(inputs="x_shape")
                self.shapes = []

            def on_batch_end(self, data: Data) -> None:
                self.shapes.append(data['x_shape'])

            def on_epoch_end(self, data: Data) -> None:
                data.write_with_log(key="batch_shapes", value=self.shapes)

        ds1 = NumpyDataset({"x": np.ones((20, 32, 32, 1), dtype=np.float32)})
        ds2 = NumpyDataset({"x": np.ones((20, 28, 28, 1), dtype=np.float32)})

        dataset = InterleaveDataset(datasets=[ds1, ds2])
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2)
        model = fe.build(model_fn=lambda: mymodel(input_shape=(None, None, 1)), optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(inputs="x", outputs="y_pred", model=model),
            LambdaOp(inputs="x", outputs="x_shape", fn=lambda x: tf.shape(x))
        ])
        estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=1, traces=Collector())
        summary = estimator.fit("test")
        self.assertEqual(summary.history['train']['epoch'][20], 1)
        if get_num_gpus() > 1:
            # On multi-gpu machines the batch gets split apart, leading to an elongated tf.shape response
            target_32 = [1, 32, 32, 1, 1, 32, 32, 1]
            target_28 = [1, 28, 28, 1, 1, 28, 28, 1]
        else:
            target_32 = [2, 32, 32, 1]
            target_28 = [2, 28, 28, 1]
        self.assertEqual(list(summary.history['train']['batch_shapes'][20][0]), target_32)
        self.assertEqual(list(summary.history['train']['batch_shapes'][20][1]), target_28)
        self.assertEqual(list(summary.history['train']['batch_shapes'][20][2]), target_32)
        self.assertEqual(list(summary.history['train']['batch_shapes'][20][3]), target_28)

    def test_interleave_dataset_with_different_dtypes(self):
        def mymodel(input_shape):
            model = Sequential()
            model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=input_shape))
            return model

        ds1 = NumpyDataset({"x": np.ones((20, 32, 32, 1), dtype=np.float32)})
        ds2 = NumpyDataset({"x": np.ones((20, 28, 28, 1), dtype=np.float16)})

        dataset = InterleaveDataset(datasets=[ds1, ds2])
        pipeline = fe.Pipeline(train_data=dataset, batch_size=2)
        model = fe.build(model_fn=lambda: mymodel(input_shape=(None, None, 1)), optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(inputs="x", outputs="y_pred", model=model),
        ])
        estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=1)
        summary = estimator.fit("test")
        self.assertEqual(summary.history['train']['epoch'][20], 1)
