# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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

import cv2
import numpy as np

from fastestimator.op import NumpyOp


class ImageReader(NumpyOp):
    """Class for reading png or jpg images
    Args:
        parent_path (str): Parent path that will be added on given path
        grey_scale (bool): Boolean to indicate whether or not to read image as grayscale
    """
    def __init__(self, inputs=None, outputs=None, mode=None, parent_path="", grey_scale=False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.parent_path = parent_path
        self.color_flag = cv2.IMREAD_COLOR
        self.grey_scale = grey_scale
        if grey_scale:
            self.color_flag = cv2.IMREAD_GRAYSCALE

    def forward(self, path, state):
        """Reads numpy array from image path
        Args:
            path: path of the image
            state: A dictionary containing background information such as 'mode'
        Returns:
           Image as numpy array
        """
        path = os.path.normpath(os.path.join(self.parent_path, path))
        data = cv2.imread(path, self.color_flag)
        if not self.grey_scale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if not isinstance(data, np.ndarray):
            raise ValueError('cv2 did not read correctly for file "{}"'.format(path))
        if self.grey_scale:
            data = np.expand_dims(data, -1)
        return data
