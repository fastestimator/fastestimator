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

from albumentations.augmentations.transforms import Posterize as PosterizeAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class Posterize(ImageOnlyAlbumentation):
    """Reduce the number of bits for each color channel

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        num_bits: Number of high bits. If num_bits is a single value, the range will be [num_bits, num_bits]. A triplet
            of ints will be interpreted as [r, g, b], and a triplet of pairs as [[r1, r1], [g1, g2], [b1, b2]]. Must be
            in the range [0, 8].

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 num_bits: Union[int,
                                 Tuple[int, int],
                                 Tuple[int, int, int],
                                 Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = 4):
        super().__init__(PosterizeAlb(num_bits=num_bits, always_apply=True), inputs=inputs, outputs=outputs, mode=mode)
