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
from typing import Union, List, Iterable, Callable, Optional

from albumentations.augmentations.transforms import CoarseDropout as CoarseDropoutAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class CoarseDropout(ImageOnlyAlbumentation):
    """Drop rectangular regions from an image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            max_holes: Maximum number of regions to zero out.
            max_height: Maximum height of the hole.
            max_width: Maximum width of the hole.
            min_holes (int): Minimum number of regions to zero out. If `None`,
                `min_holes` is be set to `max_holes`. Default: `None`.
            min_height (int): Minimum height of the hole. Default: None. If `None`,
                `min_height` is set to `max_height`. Default: `None`.
            min_width (int): Minimum width of the hole. If `None`, `min_height` is
                set to `max_width`. Default: `None`.
            fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 max_holes: int = 8,
                 max_height: int = 8,
                 max_width: int = 8,
                 min_holes: Optional[int] = None,
                 min_height: Optional[int] = None,
                 min_width: Optional[int] = None,
                 fill_value: Union[int, float, List[int], List[float]] = 0):
        super().__init__(
            CoarseDropoutAlb(max_holes=max_holes,
                             max_height=max_height,
                             max_width=max_width,
                             min_holes=min_holes,
                             min_height=min_height,
                             min_width=min_width,
                             fill_value=fill_value,
                             always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
