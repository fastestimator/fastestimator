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
from typing import Union, Tuple, Iterable, Callable

from albumentations.augmentations.transforms import Normalize as NormalizeAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class Normalize(ImageOnlyAlbumentation):
    """Divide pixel values by a maximum value, subtract mean per channel and divide by std per channel.

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            mean: mean values
            std: std values
            max_pixel_value: maximum possible pixel value
        Image types:
            uint8, float32
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 mean: Union[float, Tuple[float]] = (0.485, 0.456, 0.406),
                 std: Union[float, Tuple[float]] = (0.229, 0.224, 0.225),
                 max_pixel_value: float = 255.0):
        super().__init__(NormalizeAlb(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
