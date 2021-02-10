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

from fastestimator.op.tensorop.un_hadamard import UnHadamard
from fastestimator.test.unittest_util import is_equal


class TestUnHadamard(unittest.TestCase):
    def test_tf_4class(self):
        fromhadamard = UnHadamard(inputs='y', outputs='y', n_classes=4)
        fromhadamard.build('tf')
        output = fromhadamard.forward(
            data=[
                tf.constant([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                             [-1., 1., 1., 1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, np.array([3, 2, 2, 2, 0])))

    def test_tf_10class_16code(self):
        fromhadamard = UnHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        fromhadamard.build('tf')
        output = fromhadamard.forward(
            data=[
                tf.constant([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                              1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                    -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                           1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1.]])
            ],
            state={})[0]
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, np.array([0, 1, 9, 0, 4])))

    def test_torch_4class(self):
        fromhadamard = UnHadamard(inputs='y', outputs='y', n_classes=4)
        fromhadamard.build('torch', "cuda:0" if torch.cuda.is_available() else "cpu")
        output = fromhadamard.forward(
            data=[
                torch.tensor([[1., -1., -1., 1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.], [-1., 1., -1., -1.],
                              [-1., 1., 1., 1.]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
            ],
            state={})[0]
        if torch.cuda.is_available():
            output = output.to("cpu")
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, torch.tensor([3, 2, 2, 2, 0])))

    def test_torch_10class_16code(self):
        fromhadamard = UnHadamard(inputs='y', outputs='y', n_classes=10, code_length=16)
        fromhadamard.build('torch', "cuda:0" if torch.cuda.is_available() else "cpu")
        output = fromhadamard.forward(
            data=[
                torch.tensor([[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                               1.], [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                     -1.], [1., -1., 1., -1., 1., -1., 1., -1., -1., 1., -1., 1., -1., 1., -1.,
                                            1.], [-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                              [-1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1.,
                               -1.]]).to("cuda:0" if torch.cuda.is_available() else "cpu")
            ],
            state={})[0]
        if torch.cuda.is_available():
            output = output.to("cpu")
        output = np.argmax(output, axis=-1)
        self.assertTrue(is_equal(output, torch.tensor([0, 1, 9, 0, 4])))
