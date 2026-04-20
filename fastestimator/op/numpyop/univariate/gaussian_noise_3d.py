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
class GaussianNoise3D(NumpyOp):
    """Apply additive Gaussian noise to a 3D volume.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies random Gaussian noise to the input,
    which is a standard augmentation in medical imaging for improving model robustness to noisy scans.

    Args:
        inputs: Key(s) of 3D volumes to be modified.
        outputs: Key(s) into which to write the modified volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        mean: Mean of the Gaussian noise.
        std_range: The range for the standard deviation of the noise. The actual std will be uniformly sampled from
            this range. If a single float, the range will be (0, std_range).

    Volume types:
        float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mean: float = 0.0,
                 std_range: Union[float, Tuple[float, float]] = (0.0, 0.1),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.mean = mean
        if isinstance(std_range, (int, float)):
            self.std_range = (0.0, abs(std_range))
        else:
            self.std_range = tuple(std_range)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        if std < 1e-8:
            return data
        return [self._apply(elem, std) for elem in data]

    def _apply(self, data: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(self.mean, std, size=data.shape).astype(np.float32)
        return (data + noise).astype(np.float32)
