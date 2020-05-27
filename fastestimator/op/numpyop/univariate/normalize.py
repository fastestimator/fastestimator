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

from albumentations.augmentations.transforms import Normalize as NormalizeAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class Normalize(ImageOnlyAlbumentation):
    """Divide pixel values by a maximum value, subtract mean per channel and divide by std per channel.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        mean: Mean values to subtract.
        std: The divisor.
        max_pixel_value: Maximum possible pixel value.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 mean: Union[float, Tuple[float, ...]] = (0.485, 0.456, 0.406),
                 std: Union[float, Tuple[float, ...]] = (0.229, 0.224, 0.225),
                 max_pixel_value: float = 255.0):
        super().__init__(NormalizeAlb(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
