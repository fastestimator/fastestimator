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

from albumentations.augmentations.transforms import GaussianBlur as GaussianBlurAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class GaussianBlur(ImageOnlyAlbumentation):
    """Blur the image with a Gaussian filter of random kernel size

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            blur_limit: maximum Gaussian kernel size for blurring the input image. Should be odd and in range [3, inf). 
                Default: (3, 7). If image is a float type then only 3 and 5 are valid sizes. 
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 blur_limit: Union[int, Tuple[int, int]] = 7):
        super().__init__(GaussianBlurAlb(blur_limit=blur_limit, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
