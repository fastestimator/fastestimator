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
from typing import Union, Iterable, Callable, Optional

from albumentations.augmentations.transforms import ToFloat as ToFloatAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class ToFloat(ImageOnlyAlbumentation):
    """Divides an input by max_value to give a float image in range [0,1]

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            max_value: The maximum value to serve as the divisor. If None it will be inferred by dtype. 
        Image types:
            any type
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 max_value: Optional[float] = None):
        super().__init__(ToFloatAlb(max_value=max_value, always_apply=True), inputs=inputs, outputs=outputs, mode=mode)
