# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
from typing import Iterable, List, Optional, Tuple, Union

# from albumentations.imgaug.transforms import IAACropAndPad as IAACropAndPadAlb
from albumentations.augmentations.crops import CropAndPad as CropAndPadAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class CropAndPad(MultiVariateAlbumentation):
    """Crop and pad images by pixel amounts or fractions of image sizes.

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
        px: The number of pixels to crop (negative values) or pad (positive values) on each side of the image.
            Either this or the parameter percent may be set, not both at the same time.
        percent: The number of pixels to crop (negative values) or pad (positive values) on each side of the image given
             as a fraction of the image height/width.
        pad_mode: OpenCV border mode.
        pad_cval: The constant value to use if the pad mode is BORDER_CONSTANT.  If a tuple of two number s and at least
             one of them is a float, then a random number will be uniformly sampled per image from the continuous
             interval [a, b] and used as the value. If both number s are int s, the interval is discrete. If a list of
             number, then a random value will be chosen from the elements of the list and used as the value.
        pad_cval_mask: Same as pad_cval but only for masks.
        interpolation: flag that is used to specify the interpolation algorithm
        keep_size: Whether to keep the same size as original image.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 px: Union[None, int, Tuple[int, int]] = None,
                 percent: Union[None, float, Tuple[float, float]] = None,
                 pad_mode: Union[int, str] = 'constant',
                 pad_cval: Union[int, Tuple[float], List[int]] = 0,
                 pad_cval_mask: Union[None, int, Tuple[float], List[int]] = None,
                 keep_size: bool = True,
                 interpolation: str = 'bilinear',
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[Iterable[str]] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[Iterable[str]] = None):
        order = {'nearest_neighbor': 0, 'bilinear': 1, 'bicubic': 3, 'biquartic': 4, 'biquintic': 5}[interpolation]
        border = {'constant':0, 'reflect':1}[pad_mode]
        if not pad_cval_mask:
            pad_cval_mask = pad_cval
        super().__init__(
            CropAndPadAlb(px=px,
                             percent=percent,
                             pad_mode=border,
                             pad_cval=pad_cval,
                             pad_cval_mask=pad_cval_mask,
                             keep_size=keep_size,
                             interpolation=order,
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
