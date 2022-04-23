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
from fastestimator.util.base_util import param_to_range


@traceable()
class Contrast(NumpyOp):
    """Randomly change the contrast of an image.

    This is a wrapper for functionality provided by the PIL library:
    https://github.com/python-pillow/Pillow/tree/master/src/PIL.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        limit: Factor range for changing contrast. If limit is a single float, the range will be (-limit, limit).
            A factor of 0.0 gives a solid grey image and a factor of 1.0 gives the original image.

    Image types:
        uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 limit: float = 0.54):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.limit = param_to_range(limit)
        self.in_list, self.out_list = True, True

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """
        param_mid = (self.limit[1] + self.limit[0]) / 2
        param_extent = magnitude_coef * ((self.limit[1] - self.limit[0]) / 2)
        self.limit = (param_mid - param_extent, param_mid + param_extent)

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        factor = 1.0 + random.uniform(self.limit[0], self.limit[1])
        return [Contrast._apply_contrast(elem, factor) for elem in data]

    @staticmethod
    def _apply_contrast(data: np.ndarray, factor: float) -> np.ndarray:
        im = Image.fromarray(data)
        im = ImageEnhance.Contrast(im).enhance(factor)
        return np.array(im)
