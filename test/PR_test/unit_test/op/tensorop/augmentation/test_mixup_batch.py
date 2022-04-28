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
import torch

from fastestimator.op.tensorop.augmentation.mixup_batch import MixUpBatch
from fastestimator.test.unittest_util import is_equal


class MyTFBeta:
    @staticmethod
    def sample(sample_shape=(1, )):
        return 0.5 * tf.ones(shape=sample_shape)


class MyTorchBeta:
    @staticmethod
    def sample(sample_shape=(1, )):
        return 0.5 * torch.ones(size=sample_shape)


class TestMixUpBatch(unittest.TestCase):
    def test_tf_shared_beta(self):
        data_x = tf.stack([tf.ones((32, 32, 3)), tf.zeros((32, 32, 3))])
        data_y = tf.stack([tf.ones((10)), tf.zeros((10))])
        expected_x = tf.stack([0.5 * tf.ones((32, 32, 3)), 0.5 * tf.ones((32, 32, 3))])
        expected_y = tf.stack([0.5 * tf.ones((10)), 0.5 * tf.ones((10))])
        mu = MixUpBatch(inputs=["x", "y"], outputs=["x", "y"], alpha=1.0, mode="train", shared_beta=True)
        mu.build('tf')
        mu.beta = MyTFBeta()
        output = mu.forward(data=[data_x, data_y], state={})
        self.assertTrue(is_equal(output[0], expected_x))
        self.assertTrue(is_equal(output[1], expected_y))

    def test_tf_different_beta(self):
        data_x = tf.stack([tf.ones((32, 32, 3)), tf.zeros((32, 32, 3))])
        data_y = tf.stack([tf.ones((10)), tf.zeros((10))])
        expected_x = tf.stack([0.5 * tf.ones((32, 32, 3)), 0.5 * tf.ones((32, 32, 3))])
        expected_y = tf.stack([0.5 * tf.ones((10)), 0.5 * tf.ones((10))])
        mu = MixUpBatch(inputs=["x", "y"], outputs=["x", "y"], alpha=1.0, mode="train", shared_beta=False)
        mu.build('tf')
        mu.beta = MyTFBeta()
        output = mu.forward(data=[data_x, data_y], state={})
        self.assertTrue(is_equal(output[0], expected_x))
        self.assertTrue(is_equal(output[1], expected_y))

    def test_torch_shared_beta(self):
        data_x = torch.cat([torch.ones((1, 3, 32, 32)), torch.zeros((1, 3, 32, 32))], dim=0)
        data_y = torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))], dim=0)
        expected_x = torch.cat([0.5 * torch.ones((1, 3, 32, 32)), 0.5 * torch.ones((1, 3, 32, 32))], dim=0)
        expected_y = torch.cat([0.5 * torch.ones((1, 10)), 0.5 * torch.ones((1, 10))], dim=0)
        mu = MixUpBatch(inputs=["x", "y"], outputs=["x", "y"], alpha=1.0, mode="train", shared_beta=True)
        mu.build('torch')
        mu.beta = MyTorchBeta()
        output = mu.forward(data=[data_x, data_y], state={})
        self.assertTrue(is_equal(output[0], expected_x))
        self.assertTrue(is_equal(output[1], expected_y))

    def test_torch_different_beta(self):
        data_x = torch.cat([torch.ones((1, 3, 32, 32)), torch.zeros((1, 3, 32, 32))], dim=0)
        data_y = torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))], dim=0)
        expected_x = torch.cat([0.5 * torch.ones((1, 3, 32, 32)), 0.5 * torch.ones((1, 3, 32, 32))], dim=0)
        expected_y = torch.cat([0.5 * torch.ones((1, 10)), 0.5 * torch.ones((1, 10))], dim=0)
        mu = MixUpBatch(inputs=["x", "y"], outputs=["x", "y"], alpha=1.0, mode="train", shared_beta=False)
        mu.build('torch')
        mu.beta = MyTorchBeta()
        output = mu.forward(data=[data_x, data_y], state={})
        self.assertTrue(is_equal(output[0], expected_x))
        self.assertTrue(is_equal(output[1], expected_y))
