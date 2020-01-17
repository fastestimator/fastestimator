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

from albumentations.augmentations.transforms import HueSaturationValue as HueSaturationValueAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class HueSaturationValue(ImageOnlyAlbumentation):
    """Randomly modify the hue, saturation, and value of an image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            hue_shift_limit: range for changing hue. If hue_shift_limit is a single int, the range
                will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
            sat_shift_limit: range for changing saturation. If sat_shift_limit is a single int,
                the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
            val_shift_limit: range for changing value. If val_shift_limit is a single int, the range
                will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 hue_shift_limit: Union[int, Tuple[int, int]] = 20,
                 sat_shift_limit: Union[int, Tuple[int, int]] = 30,
                 val_shift_limit: Union[int, Tuple[int, int]] = 20):
        super().__init__(
            HueSaturationValueAlb(hue_shift_limit=hue_shift_limit,
                                  sat_shift_limit=sat_shift_limit,
                                  val_shift_limit=val_shift_limit,
                                  always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
