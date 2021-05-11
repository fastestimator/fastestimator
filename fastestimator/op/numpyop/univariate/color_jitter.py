# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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

from albumentations.augmentations.transforms import ColorJitter as ColorJitterAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class ColorJitter(ImageOnlyAlbumentation):
    """Randomly changes the brightness, contrast, saturation and hue of an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        brightness: How much to jitter brightness. brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast. contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
        saturation: How much to jitter saturation. saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 brightness: Union[float, Tuple[float]] = 0.2,
                 contrast: Union[float, Tuple[float]] = 0.2,
                 saturation: Union[float, Tuple[float]] = 0.2,
                 hue: Union[float, Tuple[float]] = 0.2):
        super().__init__(
            ColorJitterAlb(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
