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
from typing import List, Optional, Union

import cv2
from albumentations.augmentations.transforms import ElasticTransform as ElasticTransformAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class ElasticTransform(MultiVariateAlbumentation):
    """Elastic deformation of images.

    Args:
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The key of masks to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The key to write the modified masks (defaults to `masks_in` if None).
        alpha: Scaling factor during point translation.
        sigma: Gaussian filter parameter. The effect (small to large) is: random -> elastic -> affine -> translation.
        alpha_affine: The range will be (-alpha_affine, alpha_affine).
        interpolation: Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        border_mode: Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        value: Padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value: Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate: Whether to smooth displacement map with fixed kernel size. Enabling this option gives ~2X
            speedup on large (512x512) images.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 alpha: float = 34.0,
                 sigma: float = 4.0,
                 alpha_affine: float = 50.0,
                 interpolation: int = cv2.INTER_LINEAR,
                 border_mode: int = cv2.BORDER_REFLECT_101,
                 value: Union[None, int, float, List[int], List[float]] = None,
                 mask_value: Union[None, int, float, List[int], List[float]] = None,
                 approximate: bool = False,
                 mode: Optional[str] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None):
        super().__init__(
            ElasticTransformAlb(alpha=alpha,
                                sigma=sigma,
                                alpha_affine=alpha_affine,
                                interpolation=interpolation,
                                border_mode=border_mode,
                                value=value,
                                mask_value=mask_value,
                                approximate=approximate,
                                always_apply=True),
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
            mode=mode)
