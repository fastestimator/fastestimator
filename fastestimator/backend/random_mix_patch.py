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
from typing import List, TypeVar, Union

import tensorflow as tf
import torch

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def random_mix_patch(tensor: Tensor, lam: Tensor, uniform_sample: Tensor) -> Tensor:
    """Randomly cut the patches from input images.

    Args:
        tensor: The input value.
        lam: Generated mixed sample.
        uniform_sample: Sample drawn from the uniform distribution

    Returns:
        The X and Y coordinates of the patch along with width and height

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    if tf.is_tensor(tensor):
        _, height, width, _ = tensor.shape
        cut_x = width * uniform_sample
        cut_y = height * uniform_sample
        cut_w = width * tf.sqrt(1 - lam)
        cut_h = height * tf.sqrt(1 - lam)
        bbox_x1 = tf.dtypes.cast(tf.round(tf.math.maximum(cut_x - cut_w / 2, 0)), tf.int32)
        bbox_x2 = tf.dtypes.cast(tf.round(tf.math.minimum(cut_x + cut_w / 2, width)), tf.int32)
        bbox_y1 = tf.dtypes.cast(tf.round(tf.math.maximum(cut_y - cut_h / 2, 0)), tf.int32)
        bbox_y2 = tf.dtypes.cast(tf.round(tf.math.minimum(cut_y + cut_h / 2, height)), tf.int32)

        return bbox_x1, bbox_x2, bbox_y1, bbox_y2, width, height
    elif isinstance(tensor, torch.Tensor):
        _, _, height, width = tensor.shape
        cut_x = width * uniform_sample
        cut_y = height * uniform_sample
        cut_w = width * torch.sqrt(1 - lam)
        cut_h = height * torch.sqrt(1 - lam)
        bbox_x1 = torch.round(torch.clamp(cut_x - cut_w / 2, min=0)).type(torch.int32)
        bbox_x2 = torch.round(torch.clamp(cut_x + cut_w / 2, max=width)).type(torch.int32)
        bbox_y1 = torch.round(torch.clamp(cut_y - cut_h / 2, min=0)).type(torch.int32)
        bbox_y2 = torch.round(torch.clamp(cut_y + cut_h / 2, max=height)).type(torch.int32)
        return bbox_x1, bbox_x2, bbox_y1, bbox_y2, width, height
    else:
        raise ValueError("Unrecognized tensor type {}".format(type(tensor)))
