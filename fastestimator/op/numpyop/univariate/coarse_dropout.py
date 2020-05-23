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
from typing import Callable, Iterable, List, Optional, Union

from albumentations.augmentations.transforms import CoarseDropout as CoarseDropoutAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class CoarseDropout(ImageOnlyAlbumentation):
    """Drop rectangular regions from an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        max_holes: Maximum number of regions to zero out.
        max_height: Maximum height of the hole.
        max_width: Maximum width of the hole.
        min_holes: Minimum number of regions to zero out. If `None`, `min_holes` is set to `max_holes`.
        min_height: Minimum height of the hole. If `None`, `min_height` is set to `max_height`.
        min_width: Minimum width of the hole. If `None`, `min_height` is set to `max_width`.
        fill_value: value for dropped pixels.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
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
