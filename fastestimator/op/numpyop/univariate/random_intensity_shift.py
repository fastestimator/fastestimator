# Copyright 2026 The FastEstimator Authors. All Rights Reserved.
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
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomIntensityShift(NumpyOp):
    """Randomly shift and scale image intensity values.

    This augmentation is commonly used in medical imaging to simulate variations in acquisition parameters such as
    scanner calibration differences, contrast agent concentration, and tissue property variation. The transform
    applies: output = data * scale + shift.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        shift_limit: The range for the additive intensity shift. If a single float, the range will be
            (-shift_limit, shift_limit).
        scale_limit: The range for the multiplicative intensity scale factor. If a single float, the range will be
            (1 - scale_limit, 1 + scale_limit).

    Image types:
        float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 shift_limit: Union[float, Tuple[float, float]] = 0.1,
                 scale_limit: Union[float, Tuple[float, float]] = 0.1,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        if isinstance(shift_limit, (int, float)):
            self.shift_limit = (-abs(shift_limit), abs(shift_limit))
        else:
            self.shift_limit = tuple(shift_limit)
        if isinstance(scale_limit, (int, float)):
            self.scale_limit = (1.0 - abs(scale_limit), 1.0 + abs(scale_limit))
        else:
            self.scale_limit = tuple(scale_limit)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        shift = np.random.uniform(self.shift_limit[0], self.shift_limit[1])
        scale = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        return [(elem * scale + shift).astype(np.float32) for elem in data]
