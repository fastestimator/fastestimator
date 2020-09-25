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

import tensorflow as tf

from fastestimator.op.tensorop.meta import Fuse
from fastestimator.op.tensorop import LambdaOp


class TestFuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_shape = (28, 28, 3)
        cls.multi_input_tf = [tf.random.uniform(shape=(28, 28, 3)), tf.random.uniform(shape=(28, 28, 3))]

    def test_single_input_tf(self):
        a = LambdaOp(inputs='x', outputs='y', fn=lambda x: x + 1, mode='test')
        b = LambdaOp(inputs=['y', 'z'], outputs='w', fn=lambda x, y: x + y, mode='test')
        fuse = Fuse([a, b])
        with self.subTest('Check op inputs'):
            self.assertListEqual(fuse.inputs, ['x', 'z'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(fuse.outputs, ['y', 'w'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(fuse.mode, {'test'})
        output = fuse.forward(data=self.multi_input_tf, state={"mode": "test", "deferred": {}})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)

    @tf.function
    def test_single_input_tf_static(self):
        a = LambdaOp(inputs='x', outputs='y', fn=lambda x: x + 1, mode='test')
        b = LambdaOp(inputs=['y', 'z'], outputs='w', fn=lambda x, y: x + y, mode='test')
        fuse = Fuse([a, b])
        with self.subTest('Check op inputs'):
            self.assertListEqual(fuse.inputs, ['x', 'z'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(fuse.outputs, ['y', 'w'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(fuse.mode, {'test'})
        output = fuse.forward(data=self.multi_input_tf, state={"mode": "test", "deferred": {}})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output image shape'):
            self.assertEqual(output[0].shape, self.output_shape)
