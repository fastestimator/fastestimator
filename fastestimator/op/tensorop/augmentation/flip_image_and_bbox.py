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
import tensorflow as tf

from fastestimator.op import TensorOp


class FlipImageAndBbox(TensorOp):
    """This clas flips, at 0.5 probability, image and its associated bounding boxes. The bounding box format is
    [x1, y1, width, height].

    Args:
        flip_left_right: Boolean representing whether to flip the image horizontally with a probability of 0.5. Default
            is True.
        flip_up_down: Boolean representing whether to flip the image vertically with a probability of 0.5. Defult is
            Flase.
        mode: Augmentation on 'training' data or 'evaluation' data.
    """
    def __init__(self, inputs=None, outputs=None, mode=None, flip_left_right=False, flip_up_down=False):
        super().__init__(inputs, outputs, mode)
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

    def forward(self, data, state):
        """Transforms the data with the augmentation transformation

        Args:
            data: Data to be transformed
            state: Information about the current execution context

        Returns:
            Flipped data.
        """
        img, x1, y1, w, h = data

        # left-right
        if tf.logical_and(self.flip_left_right, tf.greater(tf.random.uniform([], minval=0, maxval=1), 0.5)):
            img = tf.image.flip_left_right(img)
            x1 = tf.cast(img.shape[1], dtype=tf.float32) - x1 - w

        # up-down
        if tf.logical_and(self.flip_up_down, tf.greater(tf.random.uniform([], minval=0, maxval=1), 0.5)):
            img = tf.image.flip_up_down(img)
            y1 = tf.cast(img.shape[0], dtype=tf.float32) - y1 - h

        return img, x1, y1, w, h
