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
"""Flip image and its associated bounding boxes."""
import cv2
import numpy as np
import tensorflow as tf

from fastestimator.op import NumpyOp


class FlipImageAndBbox(NumpyOp):
    """This clas flips, at 0.5 probability, image and its associated bounding boxes. The bounding box format is
    [x1, y1, width, height].

    Args:
        flip_left_right: Boolean representing whether to flip the image horizontally with a probability of 0.5. Default
            is True.
        flip_up_down: Boolean representing whether to flip the image vertically with a probability of 0.5. Defult is
            Flase.
        mode: Augmentation on 'training' data or 'evaluation' data.
    """
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs, outputs, mode)

    def forward(self, data, state):
        """Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed
            state: Information about the current execution context

        Returns:
            Flipped data.
        """
        img, x1, y1, w, h, obj_label, id = data

        img_flipped = cv2.flip(img, 1)
        x1_flipped = img.shape[1] - x1 - w

        augmented_data = [
            np.array([img, img_flipped]),
            np.array([x1, x1_flipped]),
            np.array([y1, y1]),
            np.array([w, w]),
            np.array([h, h]),
            np.array([obj_label, obj_label]),
            np.array([id, id])
        ]

        return augmented_data
