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

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomCrop3D(NumpyOp):
    """Randomly crop a 3D volume to the specified size.

    This op expects input data with shape (D, H, W) or (D, H, W, C). It applies the same crop location to all inputs
    (image and mask together), which is critical for medical image segmentation.

    If the input volume is smaller than the target crop size along any dimension, that dimension will be zero-padded
    symmetrically before cropping.

    Args:
        inputs: Key(s) of 3D volumes to be cropped.
        outputs: Key(s) into which to write the cropped volumes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        crop_size: Target crop size as (D, H, W).
        pad_value: Value used for padding if the volume is smaller than crop_size.

    Volume types:
        float32, float64, int16, int32, uint8
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 crop_size: Sequence[int] = (64, 64, 64),
                 pad_value: float = 0.0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        assert len(crop_size) == 3, "crop_size must have exactly 3 elements (D, H, W)"
        self.crop_size = tuple(crop_size)
        self.pad_value = pad_value
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        # All inputs must share the first 3 spatial dimensions
        vol_shape = data[0].shape[:3]

        # Pad if necessary
        pad_needed = False
        pad_widths = []
        for dim in range(3):
            if vol_shape[dim] < self.crop_size[dim]:
                pad_needed = True
                deficit = self.crop_size[dim] - vol_shape[dim]
                pad_before = deficit // 2
                pad_after = deficit - pad_before
                pad_widths.append((pad_before, pad_after))
            else:
                pad_widths.append((0, 0))

        if pad_needed:
            data = [self._pad(elem, pad_widths) for elem in data]
            vol_shape = data[0].shape[:3]

        # Random crop origin
        starts = []
        for dim in range(3):
            max_start = vol_shape[dim] - self.crop_size[dim]
            starts.append(np.random.randint(0, max_start + 1))

        return [self._crop(elem, starts) for elem in data]

    def _pad(self, data: np.ndarray, pad_widths: List[Tuple[int, int]]) -> np.ndarray:
        if data.ndim == 4:
            pad_widths_full = pad_widths + [(0, 0)]
        else:
            pad_widths_full = pad_widths
        return np.pad(data, pad_widths_full, mode='constant', constant_values=self.pad_value)

    def _crop(self, data: np.ndarray, starts: List[int]) -> np.ndarray:
        d, h, w = starts
        cd, ch, cw = self.crop_size
        return data[d:d + cd, h:h + ch, w:w + cw].copy()
