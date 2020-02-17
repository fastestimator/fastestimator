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
"""NumpyOp for image resize."""
import cv2
import numpy as np

from fastestimator.op import NumpyOp


class Resize(NumpyOp):
    """Resize image.

    Args:
        target_size (tuple): Target image size in (height, width) format.
        resize_method (string): `bilinear`, `nearest`, `area`, and `lanczos4` are available.
        keep_ratio (bool): If `True`, the resulting image will be padded to keep the original aspect ratio.

    Returns:
        Resized `np.ndarray`.
    """
    def __init__(self, target_size, resize_method='bilinear', keep_ratio=False, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.target_size = target_size
        self.resize_method = resize_method
        if resize_method == "bilinear":
            self.resize_method = cv2.INTER_LINEAR
        elif resize_method == "nearest":
            self.resize_method = cv2.INTER_NEAREST
        elif resize_method == "area":
            self.resize_method = cv2.INTER_AREA
        elif resize_method == "lanczos4":
            self.resize_method = cv2.INTER_LANCZOS4
        self.keep_ratio = keep_ratio

    def forward(self, data, state):
        original_dim = data.ndim

        # Calculate and apply paddings
        if self.keep_ratio:
            original_ratio = data.shape[1] / data.shape[0]
            target_ratio = self.target_size[1] / self.target_size[0]
            if original_ratio >= target_ratio:
                pad = (data.shape[1] / target_ratio - data.shape[0]) / 2
                pad_boarder = (np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int), 0, 0)
            else:
                pad = (data.shape[0] * target_ratio - data.shape[1]) / 2
                pad_boarder = (0, 0, np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int))
            data = cv2.copyMakeBorder(data, *pad_boarder, cv2.BORDER_CONSTANT)

        # Resize padded image
        data = cv2.resize(data, (self.target_size[1], self.target_size[0]), self.resize_method)

        # Restore image dimension
        if data.ndim == original_dim - 1:
            data = np.expand_dims(data, -1)

        return data
