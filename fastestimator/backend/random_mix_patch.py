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
from typing import TypeVar, Union

import tensorflow as tf
import torch

from fastestimator.backend.cast import cast
from fastestimator.backend.clip_by_value import clip_by_value
from fastestimator.backend.get_image_dims import get_image_dims
from fastestimator.backend.tensor_round import tensor_round
from fastestimator.backend.tensor_sqrt import tensor_sqrt

Tensor = TypeVar('Tensor', tf.Tensor, torch.Tensor)


def random_mix_patch(tensor: Tensor, x: Tensor, y: Tensor, lam: Tensor) -> Tensor:
    """Randomly cut the patches from input images.

    If patches are going to be pasted in other image, combination ratio between two images is defined by `lam`. Cropping
    region indicates where to drop out from the image and `cut_x` & `cut_y` are used to calculate cropping region whose
    aspect ratio is proportional to the original image.

    Args:
        tensor: The input value.
        lam: Combination ratio between two data points.
        x: Rectangular mask X coordinate.
        y: Rectangular mask Y coordinate.

    Returns:
        The X and Y coordinates of the cropped patch along with width and height.

    Raises:
        ValueError: If `tensor` is an unacceptable data type.
    """
    _, img_height, img_width = get_image_dims(tensor)
    cut_x = img_width * x
    cut_y = img_height * y
    cut_w = img_width * tensor_sqrt(1 - lam)
    cut_h = img_height * tensor_sqrt(1 - lam)
    bbox_x1 = cast(tensor_round(clip_by_value(cut_x - cut_w / 2, min_value=0)), "int32")
    bbox_x2 = cast(tensor_round(clip_by_value(cut_x + cut_w / 2, max_value=img_width)), "int32")
    bbox_y1 = cast(tensor_round(clip_by_value(cut_y - cut_h / 2, min_value=0)), "int32")
    bbox_y2 = cast(tensor_round(clip_by_value(cut_y + cut_h / 2, max_value=img_height)), "int32")
    return bbox_x1, bbox_x2, bbox_y1, bbox_y2, img_width, img_height
