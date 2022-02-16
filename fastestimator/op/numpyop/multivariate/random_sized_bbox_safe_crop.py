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
from typing import Iterable, Optional, Union

import cv2
from albumentations import BboxParams
from albumentations.augmentations import RandomSizedBBoxSafeCrop as RandomSizedBBoxSafeCropAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomSizedBBoxSafeCrop(MultiVariateAlbumentation):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The key of masks to be modified (with the same random factors as the image).
        bbox_in: The key of a bounding box(es) to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The key to write the modified masks (defaults to `masks_in` if None).
        bbox_out: The key to write the modified bounding box(es) (defaults to `bbox_in` if None).
        bbox_params: Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').
        height: Height after crop and resize.
        width: Width after crop and resize.
        erosion_rate: Erosion rate applied on input image height before crop.
        interpolation: Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 height: int,
                 width: int,
                 erosion_rate: float = 0.0,
                 interpolation: int = cv2.INTER_LINEAR,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 bbox_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None,
                 bbox_out: Optional[str] = None,
                 bbox_params: Union[BboxParams, str, None] = None):
        super().__init__(
            RandomSizedBBoxSafeCropAlb(height=height,
                                       width=width,
                                       erosion_rate=erosion_rate,
                                       interpolation=interpolation,
                                       always_apply=True),
            image_in=image_in,
            mask_in=mask_in,
            masks_in=masks_in,
            bbox_in=bbox_in,
            keypoints_in=None,
            image_out=image_out,
            mask_out=mask_out,
            masks_out=masks_out,
            bbox_out=bbox_out,
            keypoints_out=None,
            bbox_params=bbox_params,
            keypoint_params=None,
            mode=mode,
            ds_id=ds_id)
