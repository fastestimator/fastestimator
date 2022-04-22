# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
import unittest

import numpy as np
import tensorflow as tf
import torch

from fastestimator.op.tensorop.augmentation import CutMixBatch
from fastestimator.test.unittest_util import MockBetaDistribution, MockUniformDistribution, check_img_similar, \
    img_to_rgb_array


class TestCutMixBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_tf_bbox = tf.random.uniform((4, 32, 32, 3))
        cls.tf_x = tf.constant([0.25])
        cls.tf_y = tf.constant([0.25])
        cls.tf_lam = tf.constant([0.5])
        cls.tf_output_bbox = (tf.constant([0]), tf.constant([19]), tf.constant([0]), tf.constant([19]), 32, 32)
        cls.test_torch_bbox = torch.rand((4, 3, 32, 32))
        cls.torch_x = torch.tensor([0.25])
        cls.torch_y = torch.tensor([0.25])
        cls.torch_lam = torch.tensor([0.5])
        cls.torch_output_bbox = (torch.tensor([0]), torch.tensor([19]), torch.tensor([0]), torch.tensor([19]), 32, 32)
        cls.tf_input = tf.stack([tf.ones((32, 32, 3)), tf.zeros((32, 32, 3))])
        cls.tf_input_y = tf.stack([tf.ones((10)), tf.zeros((10))])
        cls.torch_input = torch.cat([torch.ones((1, 3, 32, 32)), torch.zeros((1, 3, 32, 32))], dim=0)
        cls.torch_input_y = torch.cat([torch.ones((10)), torch.zeros((10))], dim=0)
        cls.tf_output_y = tf.stack([0.65 * tf.ones((10)), 0.35 * tf.ones((10))])
        cls.torch_output_y = torch.cat([0.65 * torch.ones((1, 10)), 0.35 * torch.ones((1, 10))], dim=0)

        cls.cutmix_output1 = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "..", "util", "resources", "test_cutmix1.png"))
        cls.cutmix_output2 = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "..", "util", "resources", "test_cutmix2.png"))

    def test_tf_bbox_coordinates(self):
        cutmix = CutMixBatch(inputs=["x", "y"], outputs=["x", "y"])
        output = cutmix._get_patch_coordinates(tensor=self.test_tf_bbox, x=self.tf_x, y=self.tf_y, lam=self.tf_lam)
        with self.subTest('Check length of output tuple'):
            self.assertEqual(len(output), 6)
        with self.subTest('Check output value'):
            self.assertEqual(output, self.tf_output_bbox)

    def test_torch_bbox_coordinates(self):
        cutmix = CutMixBatch(inputs=["x", "y"], outputs=["x", "y"])
        output = cutmix._get_patch_coordinates(tensor=self.test_torch_bbox,
                                               x=self.torch_x,
                                               y=self.torch_y,
                                               lam=self.torch_lam)
        with self.subTest('Check length of output tuple'):
            self.assertEqual(len(output), 6)
        with self.subTest('Check output value'):
            self.assertEqual(output, self.torch_output_bbox)

    def test_tf_output(self):
        cutmix = CutMixBatch(inputs=["x", "y"], outputs=["x", "y"])
        cutmix.beta = MockBetaDistribution('tf')
        cutmix.uniform = MockUniformDistribution('tf')
        mixed_images = cutmix.forward(data=[self.tf_input, self.tf_input_y], state={})
        images = mixed_images[0].numpy()
        y = mixed_images[1].numpy()
        with self.subTest('First mixed image'):
            self.assertTrue(check_img_similar(images[0], img_to_rgb_array(self.cutmix_output1)))
        with self.subTest('Second mixed image'):
            self.assertTrue(check_img_similar(images[1], img_to_rgb_array(self.cutmix_output2)))
        with self.subTest('lambda value'):
            self.assertEqual(np.round(np.float32(y), 2), self.tf_output_y)

    def test_torch_output(self):
        cutmix = CutMixBatch(inputs=["x", "y"], outputs=["x", "y"])
        cutmix.beta = MockBetaDistribution('torch')
        cutmix.uniform = MockUniformDistribution('torch')
        mixed_images = cutmix.forward(data=[self.torch_input, self.torch_input_y], state={})
        images = mixed_images[0].permute(0, 2, 3, 1)
        images = images.numpy()
        y = mixed_images[1].numpy()
        with self.subTest('First mixed image'):
            self.assertTrue(check_img_similar(images[0], img_to_rgb_array(self.cutmix_output1)))
        with self.subTest('Second mixed image'):
            self.assertTrue(check_img_similar(images[1], img_to_rgb_array(self.cutmix_output2)))
        with self.subTest('lambda value'):
            self.assertEqual((np.round(np.float32(y), 2), 2), self.torch_output_y)
