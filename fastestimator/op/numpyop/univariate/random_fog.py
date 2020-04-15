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

from albumentations.augmentations.transforms import RandomFog as RandomFogAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomFog(ImageOnlyAlbumentation):
    """Add fog to an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        fog_coef_lower: Lower limit for fog intensity coefficient. Should be in the range [0, 1].
        fog_coef_upper: Upper limit for fog intensity coefficient. Should be in the range [0, 1].
        alpha_coef: Transparency of the fog circles. Should be in the range [0, 1].

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 fog_coef_lower: float = 0.3,
                 fog_coef_upper: float = 1.0,
                 alpha_coef: float = 0.08):
        super().__init__(
            RandomFogAlb(fog_coef_lower=fog_coef_lower,
                         fog_coef_upper=fog_coef_upper,
                         alpha_coef=alpha_coef,
                         always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
