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
"""NumpyOp for image and bounding box resize."""
import cv2
import numpy as np

from fastestimator.op.numpyop.resize import Resize


class ResizeImageAndBbox(Resize):
    """Resize image and associated bounding boxes for object detection.

    Args:
        target_size (tuple): Target image size in (height, width) format.
        resize_method (string): `bilinear`, `nearest`, `area`, and `lanczos4` are available.
        keep_ratio (bool): If `True`, the resulting image will be padded to keep the original aspect ratio.
        inputs (list, optional): This list of 5 strings has to be in the order of image, x1 coordinates, y1
            coordinates, widths, and heights. For example, `['img', 'x1', 'y1', 'width', 'height']`.
        outputs (list, optional): Output feature names.
        mode (str, optional): Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always
            execute. Defaults to 'eval'.

    Returns:
        list: `[resized_image, x1, y1, widths, heights]`.

    """
    def __init__(self, target_size, resize_method='bilinear', keep_ratio=False, inputs=None, outputs=None, mode=None):
        super().__init__(target_size,
                         resize_method=resize_method,
                         keep_ratio=keep_ratio,
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)

    def forward(self, data, state):
        img, x1, y1, width, height = data
        original_dim = img.ndim

        # Calculate and apply paddings
        if self.keep_ratio:
            original_ratio = img.shape[1] / img.shape[0]
            target_ratio = self.target_size[1] / self.target_size[0]
            if original_ratio >= target_ratio:
                pad = (img.shape[1] / target_ratio - img.shape[0]) / 2
                pad_boarder = (np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int), 0, 0)
                y1 += np.ceil(pad).astype(np.int)
            else:
                pad = (img.shape[0] * target_ratio - img.shape[1]) / 2
                pad_boarder = (0, 0, np.ceil(pad).astype(np.int), np.floor(pad).astype(np.int))
                x1 += np.ceil(pad).astype(np.int)
            img = cv2.copyMakeBorder(img, *pad_boarder, cv2.BORDER_CONSTANT)

        # Resize padded image and associated bounding boxes
        img_resized = cv2.resize(img, (self.target_size[1], self.target_size[0]), self.resize_method)
        x1 = (np.array(x1) * self.target_size[1] / img.shape[1]).astype(np.float32)
        y1 = (np.array(y1) * self.target_size[0] / img.shape[0]).astype(np.float32)
        width = (np.array(width) * self.target_size[1] / img.shape[1]).astype(np.float32)
        height = (np.array(height) * self.target_size[0] / img.shape[0]).astype(np.float32)
        width = np.clip(width, 1, None)
        height = np.clip(height, 1, None)
        # Restore image dimension
        if img_resized.ndim == original_dim - 1:
            img_resized = np.expand_dims(img_resized, -1)
        return img_resized, x1, y1, width, height
