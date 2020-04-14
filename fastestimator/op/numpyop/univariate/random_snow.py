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

from albumentations.augmentations.transforms import RandomSnow as RandomSnowAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomSnow(ImageOnlyAlbumentation):
    """Bleach out some pixels to simulate snow.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        snow_point_lower: Lower bound of the amount of snow. Should be in the range [0, 1].
        snow_point_upper: Upper bound of the amount of snow. Should be in the range [0, 1].
        brightness_coeff: A larger number will lead to a more snow on the image. Should be >= 0.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 snow_point_lower: float = 0.1,
                 snow_point_upper: float = 0.3,
                 brightness_coeff: float = 2.5):
        super().__init__(
            RandomSnowAlb(snow_point_lower=snow_point_lower,
                          snow_point_upper=snow_point_upper,
                          brightness_coeff=brightness_coeff,
                          always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
