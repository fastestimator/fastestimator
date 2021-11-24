# Copyright 2020 The FastEstimator Authors. All Rights Reserved.
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

import fastestimator as fe
from fastestimator.dataset import GeneratorDataset


def inputs():
    while True:
        yield {'x': np.random.rand(16), 'y': np.random.randint(16)}


class TestBatchDataset(unittest.TestCase):
    def test_dataset_contraction(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        unpaired_ds = fe.dataset.ExtendDataset(dataset=ds1, spoof_length=5)
        pipeline = fe.Pipeline(unpaired_ds)

        self.assertEqual(len(pipeline.get_results(num_steps=10)), 5)

    def test_dataset_extension(self):
        ds1 = GeneratorDataset(generator=inputs(), samples_per_epoch=10)
        unpaired_ds = fe.dataset.ExtendDataset(dataset=ds1, spoof_length=15)
        pipeline = fe.Pipeline(unpaired_ds)

        self.assertEqual(len(pipeline.get_results(num_steps=20)), 15)
