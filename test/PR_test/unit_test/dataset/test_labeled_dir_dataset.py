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

import fastestimator as fe


class TestLabeledDirDataset(unittest.TestCase):
    def test_dataset(self):
        tmpdirname = tempfile.mkdtemp()

        a_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)
        b_tmpdirname = tempfile.TemporaryDirectory(dir=tmpdirname)

        a1 = open(os.path.join(a_tmpdirname.name, "a1.txt"), "x")
        a2 = open(os.path.join(a_tmpdirname.name, "a2.txt"), "x")

        b1 = open(os.path.join(b_tmpdirname.name, "b1.txt"), "x")
        b2 = open(os.path.join(b_tmpdirname.name, "b2.txt"), "x")

        dataset = fe.dataset.LabeledDirDataset(root_dir=tmpdirname)

        self.assertEqual(len(dataset), 4)
