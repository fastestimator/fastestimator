# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
import random
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from PIL import Image, ImageEnhance

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import param_to_range


@traceable()
class Color(NumpyOp):
    """Randomly change the color balance of an image.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        limit: Factor range for changing color balance. If limit is a single float, the range will be (-limit, limit).
            A factor of 0.0 gives a black and white image and a factor of 1.0 gives the original image.

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 limit: float = 0.54):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.limit = param_to_range(limit)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_color(elem) for elem in data]

    def _apply_color(self, data: np.ndarray) -> np.ndarray:
        im = Image.fromarray(data)
        factor = 1.0 + random.uniform(self.limit[0], self.limit[1])
        im = ImageEnhance.Color(im).enhance(factor)
        return np.array(im)
