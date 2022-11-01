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
import unittest

import numpy as np
import tensorflow as tf
import torch

import fastestimator as fe
from fastestimator.test.unittest_util import check_img_similar, fig_to_rgb_array, img_to_rgb_array


class TestShowImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.color_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_color.png")))

        cls.hw_ratio_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_height_width.png")))

        cls.bb_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_bounding_box.png")))

        cls.mixed_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_mixed.png")))

        cls.text_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_text.png")))

        cls.title_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_title.png")))

        cls.float_img_ans = img_to_rgb_array(
            os.path.abspath(os.path.join(__file__, "..", "resources", "test_show_image_check_float.png")))

    def test_show_image_color_np(self):
        img = np.zeros((90, 90, 3), dtype=np.uint8)
        img[:, 0:30, :] = np.array([255, 0, 0])
        img[:, 30:60, :] = np.array([0, 255, 0])
        img[:, 60:90, :] = np.array([0, 0, 255])

        fig = fe.util.ImageDisplay(image=img)

        # Now we can save it to a numpy array.
        obj1 = fig_to_rgb_array(fig.prepare())

        # If the target output image has changed, you can regenerate it with:
        # from PIL import Image
        # im = Image.fromarray(obj1)
        # im.save("../resources/test.png")

        obj2 = self.color_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_color_torch(self):
        img = np.zeros((90, 90, 3), dtype=np.uint8)
        img[:, 0:30, :] = np.array([255, 0, 0])
        img[:, 30:60, :] = np.array([0, 255, 0])
        img[:, 60:90, :] = np.array([0, 0, 255])
        img = torch.from_numpy(img.transpose((2, 0, 1)))

        fig = fe.util.ImageDisplay(image=img)

        obj1 = fig_to_rgb_array(fig.prepare())
        obj2 = self.color_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_color_tf(self):
        img = np.zeros((90, 90, 3), dtype=np.uint8)
        img[:, 0:30, :] = np.array([255, 0, 0])
        img[:, 30:60, :] = np.array([0, 255, 0])
        img[:, 60:90, :] = np.array([0, 0, 255])
        img = tf.convert_to_tensor(img)

        fig = fe.util.ImageDisplay(image=img)

        obj1 = fig_to_rgb_array(fig.prepare())
        obj2 = self.color_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_check_float_0_to_1_np(self):
        img = np.zeros((256, 256, 3), dtype=np.float32)
        for x in range(256):
            img[x, :, :] = x / 255

        fig = fe.util.ImageDisplay(image=img).prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.float_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_check_float_neg_1_to_1_np(self):
        img = np.zeros((256, 256, 3), dtype=np.float32)
        for x in range(256):
            img[x, :, :] = (x - 127.5) / 127.5

        fig = fe.util.ImageDisplay(image=img).prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.float_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_color_arbitrary_range_np(self):
        img = np.zeros((256, 256, 3), dtype=np.float32)
        for x in range(256):
            img[x, :, :] = x * 0.2

        fig = fe.util.ImageDisplay(image=img).prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.float_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_height_width_np(self):
        img = np.zeros((150, 100))

        fig = fe.util.ImageDisplay(image=img, color_map='gray').prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.hw_ratio_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_text_np(self):
        text = "apple"
        fig = fe.util.ImageDisplay(text=text).prepare()
        obj1 = fig_to_rgb_array(fig)
        obj2 = self.text_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_bounding_box_np(self):
        bg_img = np.zeros((150, 150))
        boxes = np.array([[0, 0, 10, 20], [10, 20, 30, 50], [40, 70, 200, 200],
                          [0, 0, 0, 0], [0, 0, -50, -30]])
        fig = fe.util.ImageDisplay(image=bg_img, color_map='gray', bboxes=boxes).prepare()
        obj1 = fig_to_rgb_array(fig)
        obj2 = self.bb_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_mixed_figure_layer_np(self):
        bg_img = np.ones((150, 150, 3), dtype=np.uint8) * 255
        boxes = np.array([[0, 0, 10, 20], [10, 20, 30, 50], [40, 70, 200, 200]])

        fig = fe.util.ImageDisplay(image=bg_img, bboxes=boxes, text="apple").prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.mixed_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))

    def test_show_image_title_np(self):
        img = np.zeros((150, 150, 3), dtype=np.uint8) * 255

        fig = fe.util.ImageDisplay(image=img, title="test title").prepare()

        obj1 = fig_to_rgb_array(fig)
        obj2 = self.title_img_ans
        self.assertTrue(check_img_similar(obj1, obj2))
