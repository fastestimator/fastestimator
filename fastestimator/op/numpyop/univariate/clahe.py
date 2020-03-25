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
from typing import Union, Iterable, Callable, Tuple

from albumentations.augmentations.transforms import CLAHE as CLAHEAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class CLAHE(ImageOnlyAlbumentation):
    """Apply contrast limited adaptive histogram equalization to the image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            clip_limit: upper threshold value for contrast limiting.
                If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
            tile_grid_size: size of grid for histogram equalization. Default: (8, 8).
        Image types:
            uint8
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 clip_limit: Union[float, Tuple[float, float]] = 4.0,
                 tile_grid_size: Tuple[int, int] = (8, 8)):
        super().__init__(CLAHEAlb(clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
