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
from scipy.ndimage import gaussian_filter

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class GaussianBlur3D(NumpyOp):
    """Apply Gaussian blurring to a 3D volume.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies Gaussian blurring using
    scipy.ndimage.gaussian_filter, which is commonly used in medical imaging to simulate different acquisition
    smoothness, reduce noise, or as part of a multi-scale analysis pipeline.

    Args:
        inputs: Key(s) of 3D volumes to be blurred.
        outputs: Key(s) into which to write the blurred volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        sigma_range: The range of sigma values for the Gaussian kernel. The actual sigma will be uniformly sampled
            from this range. If a single float, a fixed sigma will be used. Can also be a tuple of 3 ranges
            (one per spatial axis) for anisotropic blurring: ((s0_min, s0_max), (s1_min, s1_max), (s2_min, s2_max)).

    Volume types:
        float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 sigma_range: Union[float,
                                    Tuple[float, float],
                                    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = (0.5, 1.5),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        if isinstance(sigma_range, (int, float)):
            self.sigma_range = ((sigma_range, sigma_range), (sigma_range, sigma_range), (sigma_range, sigma_range))
            self.is_fixed = True
        elif len(sigma_range) == 2 and isinstance(sigma_range[0], (int, float)):
            self.sigma_range = (sigma_range, sigma_range, sigma_range)
            self.is_fixed = False
        else:
            self.sigma_range = sigma_range
            self.is_fixed = False
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        if self.is_fixed:
            sigma = tuple(r[0] for r in self.sigma_range)
        else:
            sigma = tuple(np.random.uniform(r[0], r[1]) for r in self.sigma_range)
        return [self._apply(elem, sigma) for elem in data]

    @staticmethod
    def _apply(data: np.ndarray, sigma: Tuple[float, ...]) -> np.ndarray:
        if data.ndim == 4:
            # (D, H, W, C) — blur spatial dims only, not channels
            sigma_full = sigma + (0, )
        else:
            sigma_full = sigma
        return gaussian_filter(data, sigma=sigma_full).astype(np.float32)
