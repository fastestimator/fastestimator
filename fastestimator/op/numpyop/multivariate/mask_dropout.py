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
from typing import Iterable, Optional, Tuple, Union

from albumentations.augmentations.transforms import MaskDropout as MaskDropoutAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class MaskDropout(MultiVariateAlbumentation):
    """Zero out objects from an image + mask pair.

    An image & mask augmentation that zero out mask and image regions corresponding to randomly chosen object instance
    from mask. The mask must be single-channel image, with zero values treated as background. The image can be any
    number of channels.

    Args:
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The key of masks to be modified (with the same random factors as the image).
                image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The key to write the modified masks (defaults to `masks_in` if None).
        max_objects: Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
        image_fill_value: Fill value to use when filling image.
            Can be 'inpaint' to apply in-painting (works only  for 3-channel images)
        mask_fill_value: Fill value to use when filling mask.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 max_objects: Union[int, Tuple[int, int]] = 1,
                 image_fill_value: Union[int, float, str] = 0,
                 mask_fill_value: Union[int, float] = 0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None):
        super().__init__(
            MaskDropoutAlb(max_objects=max_objects,
                           image_fill_value=image_fill_value,
                           mask_fill_value=mask_fill_value,
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
            mode=mode,
            ds_id=ds_id)
