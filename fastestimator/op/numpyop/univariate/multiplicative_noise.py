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

from albumentations.augmentations.transforms import MultiplicativeNoise as MultiplicativeNoiseAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class MultiplicativeNoise(ImageOnlyAlbumentation):
    """Multiply image with a random number or array of numbers

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            multiplier: If single float image will be multiplied to this number.
                If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
            per_channel: If `False`, same values for all channels will be used.
                If `True` use sample values for each channels. Default False.
            elementwise: If `False` multiply multiply all pixels in an image with a random value sampled once.
                If `True` Multiply image pixels with values that are pixelwise randomly sampled. Default: False.
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 multiplier: Union[float, Tuple[float, float]] = (0.9, 1.1),
                 per_channel: bool = False,
                 elementwise: bool = False):
        super().__init__(
            MultiplicativeNoiseAlb(multiplier=multiplier,
                                   per_channel=per_channel,
                                   elementwise=elementwise,
                                   always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
