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
from typing import Any, Dict, Iterable, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class CTWindow(NumpyOp):
    """Apply CT windowing (window/level) to convert raw Hounsfield Unit values into a displayable range.

    CT windowing is a fundamental preprocessing step in medical imaging that maps a range of CT attenuation values
    (in Hounsfield Units) to a display range. Different window settings highlight different tissues:
        - Lung window: window_width=1500, window_level=-600
        - Soft tissue: window_width=400, window_level=50
        - Bone window: window_width=1800, window_level=400
        - Brain window: window_width=80, window_level=40
        - Liver window: window_width=150, window_level=30

    The windowed output is normalized to [0, 1] by default, or to a user-specified range.

    Args:
        inputs: Key(s) of images (in Hounsfield Units) to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        window_width: The range of HU values to display.
        window_level: The center HU value of the window.
        output_min: The minimum output value (default 0.0).
        output_max: The maximum output value (default 1.0).

    Image types:
        float32, float64, int16, int32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 window_width: float = 400.0,
                 window_level: float = 50.0,
                 output_min: float = 0.0,
                 output_max: float = 1.0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.window_width = window_width
        self.window_level = window_level
        self.output_min = output_min
        self.output_max = output_max
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_window(elem) for elem in data]

    def _apply_window(self, data: np.ndarray) -> np.ndarray:
        lower = self.window_level - self.window_width / 2.0
        upper = self.window_level + self.window_width / 2.0
        data = np.clip(data, lower, upper)
        data = (data - lower) / (upper - lower)
        data = data * (self.output_max - self.output_min) + self.output_min
        return data.astype(np.float32)
