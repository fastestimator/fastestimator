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
from unittest.mock import patch

import numpy as np

from fastestimator.trace.io import ImageViewer
from fastestimator.util.data import Data
from fastestimator.util.img_data import BatchDisplay, GridDisplay


class TestImageViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = 0.5 * np.ones(shape=(1, 28, 28, 3))
        mask = np.zeros_like(cls.image)
        mask[0, 10:20, 10:30, :] = [1, 0, 0]
        bbox = np.array([[[3, 7, 10, 6, 'box1'], [20, 20, 8, 8, 'box2']]] * 1)
        cls.img_data = GridDisplay([BatchDisplay(text=np.ones((1, )), title="y"),
                                    BatchDisplay(image=cls.image, masks=mask, bboxes=bbox, title='x')])

    @patch("fastestimator.util.base_util.FigureFE.show")
    def test_image_on_epoch_end(self, mock_show):
        data = Data({'x': 0.5 * np.ones(shape=(1, 28, 28, 3))})
        imageviewer = ImageViewer(inputs='x')
        imageviewer.on_epoch_end(data=data)
        self.assertEqual(len(mock_show.mock_calls), 1)

    @patch("fastestimator.util.base_util.FigureFE.show")
    def test_imgdata_on_epoch_end(self, mock_show):
        data = Data({'x': self.img_data})
        imageviewer = ImageViewer(inputs='x')
        imageviewer.on_epoch_end(data=data)
        self.assertEqual(len(mock_show.mock_calls), 1)
