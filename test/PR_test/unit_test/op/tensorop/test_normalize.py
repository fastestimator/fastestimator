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
from numpy import arange, float32, array, testing

from fastestimator.op.tensorop.normalize import Normalize
from fastestimator.backend import to_tensor
import torch


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.numpy_array = arange(0.0, 27.0, 1.0, dtype=float32).reshape((1, 3, 3, 3))
        self.expected_result = array([[[[-1.6688062 , -1.5404365 , -1.4120668 ],
                                        [-1.283697  , -1.1553273 , -1.0269576 ],
                                        [-0.89858794, -0.77021825, -0.6418485 ]],
                                       [[-0.5134788 , -0.38510913, -0.2567394 ],
                                        [-0.1283697 ,  0.        ,  0.1283697 ],
                                        [ 0.2567394 ,  0.38510913,  0.5134788 ]],
                                       [[ 0.6418485 ,  0.77021825,  0.89858794],
                                        [ 1.0269576 ,  1.1553273 ,  1.283697  ],
                                        [ 1.4120668 ,  1.5404365 ,  1.6688062 ]]]], dtype=float32)
        self.expected_result_torch = array([[[[-1.6688062 , -1.283697  , -0.89858794],
                                            [-0.5134788 , -0.1283697 ,  0.2567394 ],
                                            [ 0.6418485 ,  1.0269576 ,  1.4120668 ]],
                                            [[-1.5404365 , -1.1553273 , -0.77021825],
                                            [-0.38510913,  0.        ,  0.38510913],
                                            [ 0.77021825,  1.1553273 ,  1.5404365 ]],
                                            [[-1.4120668 , -1.0269576 , -0.6418485 ],
                                            [-0.2567394 ,  0.1283697 ,  0.5134788 ],
                                            [ 0.89858794,  1.283697  ,  1.6688062 ]]]], dtype=float32)
        self.expected_result_multi = array([[[[-1.5491933 , -1.5491933 , -1.5491933 ],
                                            [-1.1618949 , -1.1618949 , -1.1618949 ],
                                            [-0.77459663, -0.77459663, -0.77459663]],
                                            [[-0.38729832, -0.38729832, -0.38729832],
                                            [ 0.        ,  0.        ,  0.        ],
                                            [ 0.38729832,  0.38729832,  0.38729832]],
                                            [[ 0.77459663,  0.77459663,  0.77459663],
                                            [ 1.1618949 ,  1.1618949 ,  1.1618949 ],
                                            [ 1.5491933 ,  1.5491933 ,  1.5491933 ]]]], dtype=float32)

    def test_normalize_tf(self):
        op = Normalize(mean=13.0, std=7.79)
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result, 2)

    def test_normalize_tf_int(self):
        op = Normalize(mean=13, std=7)
        op.build("tf")
        _ = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})

    def test_std_tf(self):
        op = Normalize(mean=13.0, std=None)
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result, 2)

    def test_mean_tf(self):
        op = Normalize(mean=None, std=7.78)
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result, 2)

    def test_tf(self):
        op = Normalize(mean=None, std=None)
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result, 2)

    def test_normalize_tf_multi(self):
        op = Normalize(mean=(12., 13., 14.), std=(7.745967, 7.745967, 7.745967))
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_multi, 2)

    def test_std_tf_multi(self):
        op = Normalize(mean=(12., 13., 14.), std=None)
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_multi, 2)

    def test_mean_tf_multi(self):
        op = Normalize(mean=None, std=(7.745967, 7.745967, 7.745967))
        op.build("tf")
        data = op.forward(data=tf.convert_to_tensor(self.numpy_array), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_multi, 2)

    def test_normalize_torch(self):
        op = Normalize(mean=13.0, std=7.79)
        op.build("torch", "cuda:0" if torch.cuda.is_available() else "cpu")
        data = op.forward(data=to_tensor(self.numpy_array, "torch").type(torch.float32), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_torch, 2)

    def test_std_torch(self):
        op = Normalize(mean=13.0, std=None)
        op.build("torch", "cuda:0" if torch.cuda.is_available() else "cpu")
        data = op.forward(data=to_tensor(self.numpy_array, "torch").type(torch.float32), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_torch, 2)

    def test_mean_torch(self):
        op = Normalize(mean=None, std=7.78)
        op.build("torch", "cuda:0" if torch.cuda.is_available() else "cpu")
        data = op.forward(data=to_tensor(self.numpy_array, "torch").type(torch.float32), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_torch, 2)

    def test_torch(self):
        op = Normalize(mean=None, std=None)
        op.build("torch", "cuda:0" if torch.cuda.is_available() else "cpu")
        data = op.forward(data=to_tensor(self.numpy_array, "torch").type(torch.float32), state={})
        testing.assert_array_almost_equal(data.numpy(), self.expected_result_torch, 2)
