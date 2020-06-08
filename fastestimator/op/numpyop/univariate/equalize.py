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
from typing import Iterable, List, Union

import numpy as np
from albumentations.augmentations.transforms import Equalize as EqualizeAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class Equalize(ImageOnlyAlbumentation):
    """Equalize the image histogram.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        eq_mode: {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels: If True, use equalization by channels separately, else convert image to YCbCr representation and
            use equalization by `Y` channel.
        mask: If given, only the pixels selected by the mask are included in the analysis. May be 1 channel or 3 channel
            array. Function signature must include `image` argument.
        mask_params: Params for mask function.

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 eq_mode: str = "cv",
                 by_channels: bool = True,
                 mask: Union[None, np.ndarray] = None,
                 mask_params: List[str] = ()):
        super().__init__(
            EqualizeAlb(mode=eq_mode, by_channels=by_channels, mask=mask, mask_params=mask_params, always_apply=True),
            inputs=inputs,
            outputs=outputs,
            mode=mode)
