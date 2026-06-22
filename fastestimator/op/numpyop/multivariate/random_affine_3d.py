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
from scipy.ndimage import affine_transform

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomAffine3D(NumpyOp):
    """Apply random affine transformation to a 3D volume.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies a random combination of rotation,
    scaling, and translation. The same affine transformation is applied to all inputs (image and mask together).

    This is one of the most versatile augmentations for 3D medical imaging as it can simulate patient positioning
    variations, scanner geometry differences, and anatomical scale variations.

    Args:
        inputs: Key(s) of 3D volumes to be transformed.
        outputs: Key(s) into which to write the transformed volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        rotate_range: Range of rotation angles in degrees for each axis. If a single float, the same range
            (-rotate_range, rotate_range) is used for all axes. Can also be a sequence of 3 floats for per-axis ranges.
        scale_range: Range of scaling factors. If a single float, the range will be (1-scale_range, 1+scale_range).
            Can also be a tuple of (min_scale, max_scale).
        translate_range: Range of translation in pixels. If a single float, the range will be
            (-translate_range, translate_range) for all axes. Can also be a sequence of 3 floats.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic). Use 0 for label masks.
        fill_value: Value used for points outside the boundaries of the input.
        label_keys: Key(s) that should be treated as label/mask data (using nearest-neighbor interpolation).

    Volume types:
        float32, float64, int16, int32, uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 rotate_range: Union[float, Sequence[float]] = 10.0,
                 scale_range: Union[float, Tuple[float, float]] = 0.1,
                 translate_range: Union[float, Sequence[float]] = 10.0,
                 order: int = 1,
                 fill_value: float = 0.0,
                 label_keys: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        if isinstance(rotate_range, (int, float)):
            self.rotate_range = [(-abs(rotate_range), abs(rotate_range))] * 3
        else:
            self.rotate_range = [(-abs(r), abs(r)) for r in rotate_range]
        if isinstance(scale_range, (int, float)):
            self.scale_range = (1.0 - abs(scale_range), 1.0 + abs(scale_range))
        else:
            self.scale_range = tuple(scale_range)
        if isinstance(translate_range, (int, float)):
            self.translate_range = [(-abs(translate_range), abs(translate_range))] * 3
        else:
            self.translate_range = [(-abs(t), abs(t)) for t in translate_range]
        self.order = order
        self.fill_value = fill_value
        self.label_keys = set() if label_keys is None else (
            {label_keys} if isinstance(label_keys, str) else set(label_keys))
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        shape = data[0].shape[:3]
        matrix, offset = self._generate_affine(shape)
        results = []
        for i, elem in enumerate(data):
            key = self.inputs[i] if i < len(self.inputs) else None
            is_label = key in self.label_keys
            results.append(self._apply(elem, matrix, offset, is_label))
        return results

    def _generate_affine(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random 3D affine transformation matrix centered on the volume."""
        # Random rotation angles in radians
        angles = [np.deg2rad(np.random.uniform(r[0], r[1])) for r in self.rotate_range]

        # Rotation matrices around each axis
        cos_a, sin_a = np.cos(angles[0]), np.sin(angles[0])
        cos_b, sin_b = np.cos(angles[1]), np.sin(angles[1])
        cos_c, sin_c = np.cos(angles[2]), np.sin(angles[2])

        # Rotation around Z (axis 0, D-axis)
        rz = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        # Rotation around Y (axis 1, H-axis)
        ry = np.array([[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 0, cos_b]])
        # Rotation around X (axis 2, W-axis)
        rx = np.array([[cos_c, -sin_c, 0], [sin_c, cos_c, 0], [0, 0, 1]])

        rotation = rz @ ry @ rx

        # Random scale
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scale_matrix = np.diag([scale, scale, scale])

        # Combined rotation + scale
        matrix = rotation @ scale_matrix

        # Center of volume
        center = np.array([s / 2.0 for s in shape])

        # Random translation
        translate = np.array([np.random.uniform(t[0], t[1]) for t in self.translate_range])

        # Compute offset so that rotation is around the center
        offset = center - matrix @ center + translate

        return matrix, offset

    def _apply(self, data: np.ndarray, matrix: np.ndarray, offset: np.ndarray, is_label: bool) -> np.ndarray:
        order = 0 if is_label else self.order
        if data.ndim == 3:
            result = affine_transform(data, matrix, offset=offset, order=order, mode='constant', cval=self.fill_value)
        else:
            # (D, H, W, C): apply affine to each channel independently
            channels = []
            for c in range(data.shape[3]):
                ch = affine_transform(data[..., c],
                                      matrix,
                                      offset=offset,
                                      order=order,
                                      mode='constant',
                                      cval=self.fill_value)
                channels.append(ch)
            result = np.stack(channels, axis=-1)
        return result.astype(data.dtype)
