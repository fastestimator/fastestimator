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
from typing import Iterable, Tuple, Union

from albumentations.augmentations.transforms import RGBShift as RGBShiftAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class RGBShift(ImageOnlyAlbumentation):
    """Randomly shift the channel values for an input RGB image.

    Args:
        inputs: Key(s) of images to be normalized.
        outputs: Key(s) into which to write the normalized images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        r_shift_limit: range for changing values for the red channel. If r_shift_limit is a single int, the range
            will be (-r_shift_limit, r_shift_limit).
        g_shift_limit: range for changing values for the green channel. If g_shift_limit is a single int, the range
            will be (-g_shift_limit, g_shift_limit).
        b_shift_limit: range for changing values for the blue channel. If b_shift_limit is a single int, the range
            will be (-b_shift_limit, b_shift_limit).

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 r_shift_limit: Union[int, Tuple[int, int]] = 20,
                 g_shift_limit: Union[int, Tuple[int, int]] = 20,
                 b_shift_limit: Union[int, Tuple[int, int]] = 20):
        super().__init__(
            RGBShiftAlb(r_shift_limit=r_shift_limit,
                        g_shift_limit=g_shift_limit,
                        b_shift_limit=b_shift_limit,
                        always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
