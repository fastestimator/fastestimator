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
from typing import Tuple
import tensorflow as tf

from fastestimator.test.unittest_util import check_img_similar, fig_to_rgb_array, img_to_rgb_array
from fastestimator.util import BatchDisplay, GridDisplay


class TestImageData(unittest.TestCase):
    output_img: str
    input_image_shape: Tuple[int, int]
    label_shape: Tuple[int]
    x_test: tf.Tensor
    y_test: tf.Tensor
    img_data: GridDisplay

    @classmethod
    def setUpClass(cls):
        cls.output_img = os.path.abspath(os.path.join(__file__, "..", "resources", "test_img_data_paintfig.png"))
        cls.input_image_shape = (150, 150)
        cls.label_shape = (4, )
        cls.x_test = 0.5 * tf.ones((4, 150, 150, 3))
        cls.y_test = tf.ones(cls.label_shape)
        cls.img_data = GridDisplay([BatchDisplay(text=cls.y_test, title="y"),
                                    BatchDisplay(image=cls.x_test, title="x")])

    def test_img_data_batch_size(self):
        self.assertEqual(self.img_data.batch_size, 4)

    def test_paint_figure(self):
        fig = self.img_data.prepare()
        output = img_to_rgb_array(self.output_img)
        output_test = fig_to_rgb_array(fig)
        self.assertTrue(check_img_similar(output, output_test))
