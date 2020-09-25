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

from fastestimator.op.numpyop.meta import Repeat
from fastestimator.op.numpyop import LambdaOp


class TestRepeat(unittest.TestCase):
    def test_single_repeat_int(self):
        add_op = LambdaOp(inputs='x', outputs=('x', 'y'), fn=lambda x: (x + 1, x * x), mode='eval')
        repeat_op = Repeat(add_op, repeat=1)
        with self.subTest('Check op inputs'):
            self.assertListEqual(repeat_op.inputs, ['x'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(repeat_op.outputs, ['x', 'y'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(repeat_op.mode, {'eval'})
        output = repeat_op.forward(data=[np.ones([1])], state={"mode": "eval"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output value (x)'):
            self.assertEqual(2, output[0])
        with self.subTest('Check output value (y)'):
            self.assertEqual(1, output[1])

    def test_multi_repeat_int(self):
        add_op = LambdaOp(inputs='x', outputs=('x', 'y'), fn=lambda x: (x + 1, x * x), mode='eval')
        repeat_op = Repeat(add_op, repeat=5)
        with self.subTest('Check op inputs'):
            self.assertListEqual(repeat_op.inputs, ['x'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(repeat_op.outputs, ['x', 'y'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(repeat_op.mode, {'eval'})
        output = repeat_op.forward(data=[np.ones([1])], state={"mode": "eval"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output value (x)'):
            self.assertEqual(6, output[0])
        with self.subTest('Check output value (y)'):
            self.assertEqual(25, output[1])

    def test_single_repeat_fn_interior_value(self):
        add_op = LambdaOp(inputs='x', outputs=('x', 'y'), fn=lambda x: (x + 1, x * x), mode='eval')
        repeat_op = Repeat(add_op, repeat=lambda y: y < 1)
        with self.subTest('Check op inputs'):
            self.assertListEqual(repeat_op.inputs, ['x'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(repeat_op.outputs, ['x', 'y'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(repeat_op.mode, {'eval'})
        output = repeat_op.forward(data=[np.ones([1])], state={"mode": "eval"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output value (x)'):
            self.assertEqual(2, output[0])
        with self.subTest('Check output value (y)'):
            self.assertEqual(1, output[1])

    def test_multi_repeat_fn_interior_value(self):
        add_op = LambdaOp(inputs='x', outputs=('x', 'y'), fn=lambda x: (x + 1, x * x), mode='eval')
        repeat_op = Repeat(add_op, repeat=lambda y: y < 25)
        with self.subTest('Check op inputs'):
            self.assertListEqual(repeat_op.inputs, ['x'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(repeat_op.outputs, ['x', 'y'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(repeat_op.mode, {'eval'})
        output = repeat_op.forward(data=[np.ones([1])], state={"mode": "eval"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output value (x)'):
            self.assertEqual(6, output[0])
        with self.subTest('Check output value (y)'):
            self.assertEqual(25, output[1])

    def test_repeat_fn_exterior_value(self):
        add_op = LambdaOp(inputs='x', outputs=('x', 'y'), fn=lambda x: (x + 1, x * x), mode='eval')
        repeat_op = Repeat(add_op, repeat=lambda y, z: y + z < 25)
        with self.subTest('Check op inputs'):
            self.assertListEqual(repeat_op.inputs, ['x', 'z'])
        with self.subTest('Check op outputs'):
            self.assertListEqual(repeat_op.outputs, ['x', 'y'])
        with self.subTest('Check op mode'):
            self.assertSetEqual(repeat_op.mode, {'eval'})
        output = repeat_op.forward(data=[np.ones([1]), 10 + np.ones([1])], state={"mode": "eval"})
        with self.subTest('Check output type'):
            self.assertEqual(type(output), list)
        with self.subTest('Check output value (x)'):
            self.assertEqual(5, output[0])
        with self.subTest('Check output value (y)'):
            self.assertEqual(16, output[1])
