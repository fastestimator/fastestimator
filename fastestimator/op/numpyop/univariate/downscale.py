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
from typing import Callable, Iterable, Union

import cv2
from albumentations.augmentations.transforms import Downscale as DownscaleAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class Downscale(ImageOnlyAlbumentation):
    """Decrease image quality by downscaling and then upscaling.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        scale_min: Lower bound on the image scale. Should be < 1.
        scale_max:  Upper bound on the image scale. Should be >= scale_min.
        interpolation: cv2 interpolation method.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 scale_min: float = 0.25,
                 scale_max: float = 0.25,
                 interpolation: int = cv2.INTER_NEAREST):
        super().__init__(
            DownscaleAlb(scale_min=scale_min, scale_max=scale_max, interpolation=interpolation, always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
