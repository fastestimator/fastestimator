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

from albumentations.augmentations.blur.transforms import Blur as BlurAlb
from albumentations.augmentations.blur.transforms import AdvancedBlur as AdvBlurAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class Blur(ImageOnlyAlbumentation):
    """Blur the image with a randomly-sized kernel

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        blur_limit: maximum kernel size for blurring the input image. Should be in range [3, inf).
        advanced: Implements AdvancedBlur. Only when 'advanced' is set to True, parameters 'sigmaX_limit',
            'sigmaY_limit', 'rotate_limit', 'beta_limit' and 'noise_limit' are used.
        sigmaX_limit: Gaussian kernel standard deviation. Must be in range [0, inf). Used only when 'advanced' is True.
        sigmaY_limit: Gaussian kernel standard deviation. Must be in range [0, inf). Used only when 'advanced' is True.
        rotate_limit: Range from which a random angle used to rotate Gaussian kernel is picked. If limit is a single
            int an angle is picked from (-rotate_limit, rotate_limit). Used only when 'advanced' is True.
        beta_limit: Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution
            tails heavier than normal, values above 1.0 make it lighter than normal. Used only when 'advanced' is True.
        noise_limit: Multiplicative factor that control strength of kernel noise. Must be positive and preferably
            centered around 1.0. If set single value noise_limit will be in range (0, noise_limit). Used only when
            'advanced' is True

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 blur_limit: Union[int, Tuple[int, int]] = 7,
                 advanced: bool= False,
                 sigmaX_limit: Union[None, float, Tuple[float, float]] = (0.2, 1.0),
                 sigmaY_limit: Union[None, float, Tuple[float, float]] = (0.2, 1.0),
                 rotate_limit: Union[None, int, Tuple[int, int]] = 90,
                 beta_limit: Union[None, float, Tuple[float, float]] = (0.5, 8.0),
                 noise_limit: Union[None, float, Tuple[float, float]] = (0.9, 1.1)):

        if advanced:
            func = AdvBlurAlb(blur_limit=blur_limit,
                              sigmaX_limit=sigmaX_limit,
                              sigmaY_limit=sigmaY_limit,
                              rotate_limit=rotate_limit,
                              beta_limit=beta_limit,
                              noise_limit=noise_limit,
                              always_apply=True)
        else:
            func = BlurAlb(blur_limit=blur_limit,always_apply=True)

        super().__init__(func,
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)
