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

import fastestimator as fe
from fastestimator.dataset.interleave_dataset import InterleaveDataset
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op.numpyop import Batch, NumpyOp


class Plus(NumpyOp):
    def forward(self, data, state):
        return data + 0.5


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