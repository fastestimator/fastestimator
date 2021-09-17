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

from albumentations.augmentations.transforms import RandomGridShuffle as RandomGridShuffleAlb

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomGridShuffle(MultiVariateAlbumentation):
    """Divide an image into a grid and randomly shuffle the grid's cells.

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
        grid: size of grid for splitting image (height, width).

    Image types:
        uint8, float32
    """
    def __init__(self,
                 grid: Tuple[int, int] = (3, 3),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[str] = None):
        super().__init__(RandomGridShuffleAlb(grid=grid, always_apply=True),
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
