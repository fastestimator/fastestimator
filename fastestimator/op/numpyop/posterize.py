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

from albumentations.augmentations.transforms import Posterize as PosterizeAlb

from fastestimator.op.numpyop.base_augmentations import ImageOnlyAlbumentation


class Posterize(ImageOnlyAlbumentation):
    """Reduce the number of bits for each color channel 

        Args:
            inputs: Key(s) of images to be normalized
            outputs: Key(s) of images to be normalized
            mode: What execution mode (train, eval, None) to apply this operation
            num_bits ((int, int) or int, or list of ints [r, g, b],
                      or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
                If num_bits is a single value, the range will be [num_bits, num_bits].
                Must be in range [0, 8]. Default: 4.
        Image types:
            uint8
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 num_bits: Union[int,
                                 Tuple[int, int],
                                 Tuple[int, int, int],
                                 Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = 4):
        super().__init__(PosterizeAlb(num_bits=num_bits, always_apply=True), inputs=inputs, outputs=outputs, mode=mode)
