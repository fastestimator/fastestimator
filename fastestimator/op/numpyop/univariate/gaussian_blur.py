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

from albumentations.augmentations import GaussianBlur as GaussianBlurAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class GaussianBlur(ImageOnlyAlbumentation):
    """Blur the image with a Gaussian filter of random kernel size.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        blur_limit: Maximum Gaussian kernel size for blurring the input image. Should be odd and in range [3, inf).
        sigma_limit: Gaussian kernel standard deviation. Must be greater in range [0, inf). If set single value
            sigma_limit will be in range (0, sigma_limit). If set to 0 sigma will be computed as sigma =
            0.3*((ksize-1)*0.5 - 1)

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 blur_limit: Union[int, Tuple[int, int]] = 7,
                 sigma_limit: Union[float, Tuple[float, float]] = 0.0):

        super().__init__(GaussianBlurAlb(blur_limit=blur_limit, always_apply=True, sigma_limit=sigma_limit),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)
