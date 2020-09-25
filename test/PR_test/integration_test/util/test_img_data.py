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

import matplotlib
import numpy as np
import tensorflow as tf

from fastestimator.test.unittest_util import check_img_similar, fig_to_rgb_array, img_to_rgb_array
from fastestimator.util import ImgData


class TestImageData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_img = os.path.abspath(os.path.join(__file__, "..", "resources", "test_img_data_paintfig.png"))
        cls.input_image_shape = (150, 150)
        cls.label_shape = (4, )
        cls.x_test = 0.5 * tf.ones((4, 150, 150, 3))
        cls.y_test = tf.ones(cls.label_shape)
        cls.img_data = ImgData(y=cls.y_test, x=cls.x_test)

    def setUp(self) -> None:
        self.old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

    def tearDown(self) -> None:
        matplotlib.use(self.old_backend)

    def test_n_cols(self):
        self.assertEqual(self.img_data._n_cols(), 2)

    def test_n_rows(self):
        self.assertEqual(self.img_data._n_rows(), 1)

    def test_shape_to_width_1d(self):
        self.assertEqual(self.img_data._shape_to_width(self.label_shape, min_width=300),
                         300,
                         'Output should be equal to minimum width')

    def test_shape_to_width_2d(self):
        self.assertEqual(self.img_data._shape_to_width(self.input_image_shape, min_width=100),
                         150,
                         'Output should be equal to input width')

    def test_shape_to_height_1d(self):
        self.assertEqual(self.img_data._shape_to_height(self.label_shape, min_height=300),
                         300,
                         'Output should be equal to minimum height')

    def test_shape_to_height_2d(self):
        self.assertEqual(self.img_data._shape_to_height(self.input_image_shape, min_height=150),
                         150,
                         'Output should be equal to input height')

    def test_img_data_widths(self):
        index = 0
        self.assertEqual(self.img_data._widths(index), [(0, 200), (250, 450)])

    def test_img_data_total_width(self):
        self.assertEqual(self.img_data._total_width(), 450)

    def test_img_data_heights(self):
        self.assertEqual(self.img_data._heights(), [(10, 810)])

    def test_img_data_total_height(self):
        self.assertEqual(self.img_data._total_height(), 840)

    def test_img_data_batch_size(self):
        self.assertEqual(self.img_data._batch_size(0), 4)

    def test_paint_figure(self):
        fig = self.img_data.paint_figure()
        output = img_to_rgb_array(self.output_img)
        output_test = fig_to_rgb_array(fig)
        self.assertTrue(check_img_similar(output, output_test))

    def test_paint_numpy(self):
        output_test = self.img_data.paint_numpy()
        output_test = np.squeeze(output_test, axis=0)
        output = img_to_rgb_array(self.output_img)
        self.assertTrue(check_img_similar(output, output_test))
