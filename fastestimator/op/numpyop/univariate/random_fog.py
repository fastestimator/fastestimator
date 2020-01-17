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
from typing import Union, Iterable, Callable

from albumentations.augmentations.transforms import RandomFog as RandomFogAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class RandomFog(ImageOnlyAlbumentation):
    """Add fog to an image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            fog_coef_lower: lower limit for fog intensity coefficient. Should be in [0, 1] range.
            fog_coef_upper: upper limit for fog intensity coefficient. Should be in [0, 1] range.
            alpha_coef: transparency of the fog circles. Should be in [0, 1] range.
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
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
