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

from albumentations.augmentations.transforms import RandomBrightnessContrast as RandomBrightnessContrastAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomBrightnessContrast(ImageOnlyAlbumentation):
    """Randomly change the brightness and contrast of an image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            brightness_limit: factor range for changing brightness.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            contrast_limit: factor range for changing contrast.
                If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
            brightness_by_max: If True adjust contrast by image dtype maximum,
                else adjust contrast by image mean.
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 brightness_limit: Union[float, Tuple[float, float]] = 0.2,
                 contrast_limit: Union[float, Tuple[float, float]] = 0.2,
                 brightness_by_max: bool = True):
        super().__init__(
            RandomBrightnessContrastAlb(brightness_limit=brightness_limit,
                                        contrast_limit=contrast_limit,
                                        brightness_by_max=brightness_by_max,
                                        always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
