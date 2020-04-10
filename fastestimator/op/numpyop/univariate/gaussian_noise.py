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

from albumentations.augmentations.transforms import GaussNoise as GaussNoiseAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation



class GaussianNoise(ImageOnlyAlbumentation):
    """Apply gaussian noise to the image

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            var_limit: variance range for noise. If var_limit is a single float, the range will be (0, var_limit).
                Default: (10.0, 50.0).
            mean: mean of the noise. Default: 0
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 var_limit: Union[float, Tuple[float, float]] = (10.0, 50.0),
                 mean: float = 0.0):
        super().__init__(GaussNoiseAlb(var_limit=var_limit, mean=mean, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
