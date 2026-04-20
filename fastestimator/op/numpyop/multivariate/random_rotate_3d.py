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
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomRotate3D(NumpyOp):
    """Randomly rotate a 3D volume around one or more axes.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies the same random rotation to all
    inputs (image and mask together). Rotation is performed using scipy.ndimage.rotate.

    Args:
        inputs: Key(s) of 3D volumes to be rotated.
        outputs: Key(s) into which to write the rotated volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        angle_range: The range of rotation angles in degrees. If a single float, the range will be
            (-angle_range, angle_range).
        axes: A list of axis-plane tuples defining the rotation planes. Each element is a tuple of two axes
            (e.g. (0,1) rotates in the D-H plane, (1,2) in the H-W plane, (0,2) in the D-W plane).
            By default, rotation can occur in all three planes.
        order: Interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic). Use 0 for label masks.
        reshape: Whether to reshape the output to contain the full rotated volume. If False, the output has
            the same shape as the input, and parts may be cropped.
        fill_value: Value used for points outside the boundaries of the input.
        label_keys: Key(s) that should be treated as label/mask data (using nearest-neighbor interpolation).

    Volume types:
        float32, float64, int16, int32, uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 angle_range: Union[float, Tuple[float, float]] = 15.0,
                 axes: Sequence[Tuple[int, int]] = ((0, 1), (1, 2), (0, 2)),
                 order: int = 1,
                 reshape: bool = False,
                 fill_value: float = 0.0,
                 label_keys: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        if isinstance(angle_range, (int, float)):
            self.angle_range = (-abs(angle_range), abs(angle_range))
        else:
            self.angle_range = tuple(angle_range)
        self.axes = list(axes)
        self.order = order
        self.reshape = reshape
        self.fill_value = fill_value
        self.label_keys = set() if label_keys is None else (
            {label_keys} if isinstance(label_keys, str) else set(label_keys))
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        # Sample random angles for each rotation plane
        angles = {ax: np.random.uniform(self.angle_range[0], self.angle_range[1]) for ax in self.axes}
        results = []
        for i, elem in enumerate(data):
            key = self.inputs[i] if i < len(self.inputs) else None
            is_label = key in self.label_keys
            results.append(self._apply(elem, angles, is_label))
        return results

    def _apply(self, data: np.ndarray, angles: Dict[Tuple[int, int], float], is_label: bool) -> np.ndarray:
        order = 0 if is_label else self.order
        result = data.copy()
        for ax_pair, angle in angles.items():
            if abs(angle) < 1e-6:
                continue
            result = scipy_rotate(result,
                                  angle=angle,
                                  axes=ax_pair,
                                  reshape=self.reshape,
                                  order=order,
                                  mode='constant',
                                  cval=self.fill_value)
        return result.astype(data.dtype)
