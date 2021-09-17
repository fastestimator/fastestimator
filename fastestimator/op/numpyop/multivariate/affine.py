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
from typing import Iterable, List, Optional, Tuple, TypeVar, Union

from albumentations import BboxParams, KeypointParams
from albumentations.imgaug.transforms import IAAAffine

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable

Number = TypeVar('Number', int, float)


@traceable()
class Affine(MultiVariateAlbumentation):
    """Perform affine transformations on an image.

    Args:
        rotate: How much to rotate an image (in degrees). If a single value is given then images will be rotated by
                a value sampled from the range [-n, n]. If a tuple (a, b) is given then each image will be rotated
                by a value sampled from the range [a, b].
        scale: How much to scale an image (in percentage). If a single value is given then all images will be scaled
                by a value drawn from the range [1.0, n]. If a tuple (a,b) is given then each image will be scaled
                based on a value drawn from the range [a,b].
        shear: How much to shear an image (in degrees). If a single value is given then all images will be sheared
                on X and Y by two values sampled from the range [-n, n]. If a tuple (a, b) is given then images will
                be sheared on X and Y by two values randomly sampled from the range [a, b].
        translate: How much to translate an image. If a single value is given then the translation extent will be
                sampled from the range [0,n]. If a tuple (a,b) is given then the extent will be sampled from
                the range [a,b]. If integers are given then the translation will be in pixels. If a float then
                it will be as a fraction of the image size.
        border_handling: What to do in order to fill newly created pixels. Options are 'constant', 'edge',
                'symmetric', 'reflect', and 'wrap'. If a list is given, then the method will be randomly
                selected from the options in the list.
        fill_value: What pixel value to insert when border_handling is 'constant'.
        interpolation: What interpolation method to use. Options (from fast to slow) are 'nearest_neighbor',
                'bilinear', 'bicubic', 'biquartic', and 'biquintic'.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The key of masks to be modified (with the same random factors as the image).
        bbox_in: The key of a bounding box(es) to be modified (with the same random factors as the image).
        keypoints_in: The key of keypoints to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The key to write the modified masks (defaults to `masks_in` if None).
        bbox_out: The key to write the modified bounding box(es) (defaults to `bbox_in` if None).
        keypoints_out: The key to write the modified keypoints (defaults to `keypoints_in` if None).
        bbox_params: Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').
        keypoint_params: Parameters defining the type of keypoints ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa').

    Image types:
        uint8, float32
    """
    def __init__(self,
                 rotate: Union[Number, Tuple[Number, Number]] = 0,
                 scale: Union[float, Tuple[float, float]] = 1.0,
                 shear: Union[Number, Tuple[Number, Number]] = 0,
                 translate: Union[Number, Tuple[Number, Number]] = 0,
                 border_handling: Union[str, List[str]] = "reflect",
                 fill_value: Number = 0,
                 interpolation: str = "bilinear",
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 bbox_in: Optional[str] = None,
                 keypoints_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None,
                 bbox_out: Optional[str] = None,
                 keypoints_out: Optional[str] = None,
                 bbox_params: Union[BboxParams, str, None] = None,
                 keypoint_params: Union[KeypointParams, str, None] = None):
        order = {'nearest_neighbor': 0, 'bilinear': 1, 'bicubic': 3, 'biquartic': 4, 'biquintic': 5}[interpolation]
        if isinstance(translate, int) or (isinstance(translate, Tuple) and isinstance(translate[0], int)):
            func = IAAAffine(rotate=rotate,
                             scale=scale,
                             shear=shear,
                             translate_px=translate,
                             order=order,
                             cval=fill_value,
                             mode=border_handling,
                             always_apply=True)
        else:
            func = IAAAffine(rotate=rotate,
                             scale=scale,
                             shear=shear,
                             translate_percent=translate,
                             order=order,
                             cval=fill_value,
                             mode=border_handling,
                             always_apply=True)
        super().__init__(func,
                         image_in=image_in,
                         mask_in=mask_in,
                         masks_in=masks_in,
                         bbox_in=bbox_in,
                         keypoints_in=keypoints_in,
                         image_out=image_out,
                         mask_out=mask_out,
                         masks_out=masks_out,
                         bbox_out=bbox_out,
                         keypoints_out=keypoints_out,
                         bbox_params=bbox_params,
                         keypoint_params=keypoint_params,
                         mode=mode,
                         ds_id=ds_id)
