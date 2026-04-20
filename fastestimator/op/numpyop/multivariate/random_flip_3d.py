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
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomFlip3D(NumpyOp):
    """Randomly flip a 3D volume along one or more axes.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies the same random flip to all inputs
    (image and mask together), which is critical for medical image segmentation.

    Args:
        inputs: Key(s) of 3D volumes to be flipped.
        outputs: Key(s) into which to write the flipped volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        spatial_axes: Which spatial axes (0=Depth, 1=Height, 2=Width) are eligible for flipping.
            Each eligible axis will be independently flipped with probability 0.5.

    Volume types:
        float32, float64, int16, int32, uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 spatial_axes: Sequence[int] = (0, 1, 2),
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.spatial_axes = list(spatial_axes)
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        flip_axes = [ax for ax in self.spatial_axes if np.random.random() < 0.5]
        if not flip_axes:
            return data
        return [self._apply(elem, flip_axes) for elem in data]

    @staticmethod
    def _apply(data: np.ndarray, flip_axes: List[int]) -> np.ndarray:
        for ax in flip_axes:
            data = np.flip(data, axis=ax)
        return np.ascontiguousarray(data)
