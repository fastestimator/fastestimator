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
from typing import Callable, Iterable, Tuple, Union

from albumentations.augmentations.transforms import ISONoise as ISONoiseAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class ISONoise(ImageOnlyAlbumentation):
    """Apply camera sensor noise.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        color_shift: Variance range for color hue change. Measured as a fraction of 360 degree Hue angle in the HLS
            colorspace.
        intensity: Multiplicative factor that controls the strength of color and luminace noise.

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 color_shift: Tuple[float, float] = (0.01, 0.05),
                 intensity: Tuple[float, float] = (0.1, 0.5)):
        super().__init__(ISONoiseAlb(color_shift=color_shift, intensity=intensity, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
