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
from typing import Callable, Iterable, Optional, Tuple, Union

from albumentations.augmentations.transforms import RandomRain as RandomRainAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomRain(ImageOnlyAlbumentation):
    """Add rain to an image

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        slant_lower: Should be in range [-20, 20].
        slant_upper: Should be in range [-20, 20].
        drop_length: Should be in range [0, 100].
        drop_width: Should be in range [1, 5].
        drop_color: Rain lines color (r, g, b).
        blur_value: How blurry to make the rain.
        brightness_coefficient: Rainy days are usually shady. Should be in range [0, 1].
        rain_type: One of [None, "drizzle", "heavy", "torrential"].

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 slant_lower: int = -10,
                 slant_upper: int = 10,
                 drop_length: int = 20,
                 drop_width: int = 1,
                 drop_color: Tuple[int, int, int] = (200, 200, 200),
                 blur_value: int = 7,
                 brightness_coefficient: float = 0.7,
                 rain_type: Optional[str] = None):
        super().__init__(
            RandomRainAlb(slant_lower=slant_lower,
                          slant_upper=slant_upper,
                          drop_length=drop_length,
                          drop_width=drop_width,
                          drop_color=drop_color,  # Their docstring type hint doesn't match the real code
                          blur_value=blur_value,
                          brightness_coefficient=brightness_coefficient,
                          rain_type=rain_type,
                          always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
