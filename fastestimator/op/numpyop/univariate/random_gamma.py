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

from albumentations.augmentations.transforms import RandomGamma as RandomGammaAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class RandomGamma(ImageOnlyAlbumentation):
    """Apply a gamma transform to an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        gamma_limit: If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit).
        eps: A numerical stability constant to avoid division by zero.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 gamma_limit: Union[float, Tuple[float, float]] = (80, 120),
                 eps: float = 1e-7):
        super().__init__(RandomGammaAlb(gamma_limit=gamma_limit, eps=eps, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
