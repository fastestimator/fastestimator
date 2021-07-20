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
import os
import tempfile
import unittest

import numpy as np

import fastestimator as fe


def inputs():
    while True:
        yield {'x': np.random.rand(16), 'y': np.random.randint(16)}


class TestSiameseDirDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        a_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)
        b_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)

        a1 = open(os.path.join(a_tmpdirname.name, "fa1.txt"), "x")
        a2 = open(os.path.join(a_tmpdirname.name, "fa2.txt"), "x")

        b1 = open(os.path.join(b_tmpdirname.name, "fb1.txt"), "x")
        b2 = open(os.path.join(b_tmpdirname.name, "fb2.txt"), "x")

        dataset = fe.dataset.SiameseDirDataset(root_dir=tmpdirname)

        self.assertEqual(len(dataset), 4)
