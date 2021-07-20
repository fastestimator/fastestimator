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
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from fastestimator.test.unittest_util import sample_system_object
from fastestimator.trace.io import ImageSaver
from fastestimator.util.data import Data
from fastestimator.util.img_data import ImgData


class TestImageSaver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_dir = tempfile.gettempdir()
        cls.image_path = os.path.join(cls.image_dir, 'img_train_epoch_0_elem_0.png')
        cls.img_data_path = os.path.join(cls.image_dir, 'img_data_train_epoch_0.png')
        cls.input_img = 0.5 * np.ones((1, 32, 32, 3))
        cls.mask = np.zeros_like(cls.input_img)
        cls.mask[0, 10:20, 10:30, :] = [1, 0, 0]
        bbox = np.array([[[3, 7, 10, 6, 'box1'], [20, 20, 8, 8, 'box2']]] * 1)
        d = ImgData(y=np.ones((1, )), x=[cls.input_img, cls.mask, bbox])
        cls.data = Data({'img': cls.input_img, 'img_data': d})

    def test_on_epoch_end(self):
        image_saver = ImageSaver(inputs='img', save_dir=self.image_dir)
        image_saver.system = sample_system_object()
        image_saver.on_epoch_end(data=self.data)
        with self.subTest('Check if image is saved'):
            self.assertTrue(os.path.exists(self.image_path))
        with self.subTest('Check image is valid or not'):
            im = plt.imread(self.image_path)
            self.assertFalse(np.any(im[:, 0] == np.nan))

    def test_on_epoch_end_img_data(self):
        if os.path.exists(self.img_data_path):
            os.remove(self.img_data_path)
        image_saver = ImageSaver(inputs='img_data', save_dir=self.image_dir)
        image_saver.system = sample_system_object()
        image_saver.on_epoch_end(data=self.data)
        with self.subTest('Check if image is saved'):
            self.assertTrue(os.path.exists(self.img_data_path))
        with self.subTest('Check image is valid or not'):
            im = plt.imread(self.img_data_path)
            self.assertFalse(np.any(im[:, 0] == np.nan))
