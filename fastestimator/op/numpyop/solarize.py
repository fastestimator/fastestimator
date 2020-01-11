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

from albumentations.augmentations.transforms import Solarize as SolarizeAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class Solarize(ImageOnlyAlbumentation):
    """Invert all pixel values above a threshold

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            threshold ((int, int) or int, or (float, float) or float): range for solarizing threshold.
                If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 threshold: Union[int, Tuple[int, int], float, Tuple[float, float]] = 20):
        super().__init__(SolarizeAlb(threshold=threshold, always_apply=True), inputs=inputs, outputs=outputs, mode=mode)
