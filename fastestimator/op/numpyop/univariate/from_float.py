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
from typing import Callable, Iterable, Optional, Union

import numpy as np
from albumentations.augmentations.transforms import FromFloat as FromFloatAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation


class FromFloat(ImageOnlyAlbumentation):
    """Takes an input float image in range [0, 1.0] and then multiplies by `max_value` to get an int image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        max_value: The maximum value to serve as the multiplier. If None it will be inferred by dtype.
        dtype: The data type to cast the output as.

    Image types:
        float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str], Callable],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 max_value: Optional[float] = None,
                 dtype: Union[str, np.dtype] = "uint16"):
        super().__init__(FromFloatAlb(max_value=max_value, dtype=dtype, always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode)
