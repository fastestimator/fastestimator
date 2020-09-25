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
import tensorflow as tf
import torch

import fastestimator as fe
import fastestimator.test.unittest_util as fet


class TestToType(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_np = {
            "x": np.ones((10, 15), dtype="float32"),
            "y": [np.ones((4), dtype="int8"), np.ones((5, 3), dtype="double")],
            "z": {
                "key": np.ones((2, 2), dtype="int64")
            }
        }
        cls.data_tf = {
            "x": tf.ones((10, 15), dtype="float32"),
            "y": [tf.ones((4), dtype="int8"), tf.ones((5, 3), dtype="double")],
            "z": {
                "key": tf.ones((2, 2), dtype="int64")
            }
        }
        cls.data_torch = {
            "x": torch.ones((10, 15), dtype=torch.float32),
            "y": [torch.ones((4), dtype=torch.int8), torch.ones((5, 3), dtype=torch.double)],
            "z": {
                "key": torch.ones((2, 2), dtype=torch.long)
            }
        }
        cls.op_np = {
            'x': np.dtype('float16'),
            'y': [np.dtype('float16'), np.dtype('float16')],
            'z': {
                'key': np.dtype('float16')
            }
        }
        cls.op_tf = {'x': tf.uint8, 'y': [tf.uint8, tf.uint8], 'z': {'key': tf.uint8}}
        cls.op_torch = {'x': torch.float64, 'y': [torch.float64, torch.float64], 'z': {'key': torch.float64}}

    def test_to_type_np(self):
        data = fe.backend.cast(self.data_np, "float16")
        types = fe.backend.to_type(data)
        self.assertTrue(fet.is_equal(types, self.op_np))

    def test_to_type_tf(self):
        data = fe.backend.cast(self.data_tf, "uint8")
        types = fe.backend.to_type(data)
        self.assertTrue(fet.is_equal(types, self.op_tf))

    def test_to_type_torch(self):
        data = fe.backend.cast(self.data_torch, "float64")
        types = fe.backend.to_type(data)
        self.assertTrue(fet.is_equal(types, self.op_torch))
