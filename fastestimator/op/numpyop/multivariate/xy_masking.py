# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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
from typing import Iterable, Optional, Union

from albumentations.augmentations.dropout import XYMasking as XYMaskingAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class XYMasking(MultiVariateAlbumentation):
    """Mask specified XY regions from an image + mask pair.

    An image & mask augmentation that masks regions of the image and mask based on specified XY coordinates.
    The mask must be single-channel image, with zero values treated as background. The image can be any
    number of channels. From albumentations.

    Args:
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The list of mask keys to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The list of keys to write the modified masks (defaults to `masks_in` if None).
        num_masks_x: Number of masks along the x-axis.
        num_masks_y: Number of masks along the y-axis.
        mask_x_length: Length of each mask along the x-axis.
        mask_y_length: Length of each mask along the y-axis.
        fill_value: Value used to fill the masked areas in the image.
        mask_fill_value: Value used to fill the masked areas in the mask.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 num_masks_x: int = 0,
                 num_masks_y: int = 0,
                 mask_x_length: int = 0,
                 mask_y_length: int = 0,
                 fill_value: Union[int, float] = 0,
                 mask_fill_value: Union[int, float] = 0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[Iterable[str]] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[Iterable[str]] = None):
        super().__init__(
            XYMaskingAlb(num_masks_x=num_masks_x, num_masks_y=num_masks_y, mask_x_length=mask_x_length, mask_y_length=mask_y_length, fill_value=fill_value, mask_fill_value=mask_fill_value, always_apply=True),
            image_in=image_in,
            mask_in=mask_in,
            masks_in=masks_in,
            bbox_in=None,
            keypoints_in=None,
            image_out=image_out,
            mask_out=mask_out,
            masks_out=masks_out,
            bbox_out=None,
            keypoints_out=None,
            bbox_params=None,
            keypoint_params=None,
            mode=mode,
            ds_id=ds_id)