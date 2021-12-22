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

from fastestimator.op.numpyop import LambdaOp
from fastestimator.test.unittest_util import is_equal


class TestLambdaOp(unittest.TestCase):
    def test_single_input(self):
        op = LambdaOp(fn=np.sum)
        data = op.forward(data=[[1, 2, 3]], state={})
        self.assertEqual(data, 6)

    def test_multi_input(self):
        op = LambdaOp(fn=np.reshape)
        data = op.forward(data=[np.array([1, 2, 3, 4]), (2, 2)], state={})
        self.assertTrue(is_equal(data, np.array([[1, 2], [3, 4]])))

    def test_batch_forward(self):
        op = LambdaOp(fn=np.sum)
        data = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = op.forward_batch(data=[data], state={})
        self.assertEqual(result, 45)
