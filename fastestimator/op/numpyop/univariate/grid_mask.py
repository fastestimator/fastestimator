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
class GridMask(NumpyOp):
    """Apply GridMask augmentation by masking out regular grid patches from the image.

    GridMask drops regions of the input in a structured grid pattern. Unlike random erasing or CoarseDropout which
    mask random rectangular regions, GridMask creates a regular grid of masked regions across the entire image.
    This has been shown to be effective for medical image segmentation and classification, as it forces the model
    to learn from partial information while maintaining spatial structure.

    Reference: Chen et al., "GridMask Data Augmentation", 2020, https://arxiv.org/abs/2001.04086

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        d_range: Range for the grid cell size in pixels. The grid cell size will be randomly sampled from this range.
        ratio: The ratio of the masked region to the grid cell size. A ratio of 0.5 means half of each cell is masked.
        rotate_angle: Maximum rotation angle (degrees) to apply to the grid pattern for added variety.
        fill_value: Value used to fill the masked regions.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 d_range: Tuple[int, int] = (96, 224),
                 ratio: float = 0.6,
                 rotate_angle: float = 0.0,
                 fill_value: float = 0.0,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        assert 0.0 < ratio < 1.0, "ratio must be in (0, 1)"
        assert d_range[0] > 0 and d_range[1] >= d_range[0], "d_range must be positive with d_range[0] <= d_range[1]"
        self.d_range = d_range
        self.ratio = ratio
        self.rotate_angle = rotate_angle
        self.fill_value = fill_value
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        # Generate one mask to share across all inputs
        h, w = data[0].shape[:2]
        mask = self._generate_mask(h, w)
        return [self._apply(elem, mask) for elem in data]

    def _generate_mask(self, h: int, w: int) -> np.ndarray:
        d = np.random.randint(self.d_range[0], self.d_range[1] + 1)
        mask_size = int(d * self.ratio)

        # Create the mask on a padded canvas to handle rotation
        diag = int(np.ceil(np.sqrt(h * h + w * w)))
        mask = np.ones((diag, diag), dtype=np.float32)

        # Random offset for the grid
        offset_y = np.random.randint(0, d)
        offset_x = np.random.randint(0, d)

        for y in range(-d + offset_y, diag, d):
            for x in range(-d + offset_x, diag, d):
                y1 = max(0, y)
                y2 = min(diag, y + mask_size)
                x1 = max(0, x)
                x2 = min(diag, x + mask_size)
                if y1 < y2 and x1 < x2:
                    mask[y1:y2, x1:x2] = 0

        # Apply rotation if needed
        if self.rotate_angle > 0:
            import cv2
            angle = np.random.uniform(-self.rotate_angle, self.rotate_angle)
            center = (diag // 2, diag // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=1.0)

        # Crop to image size from the center
        start_y = (diag - h) // 2
        start_x = (diag - w) // 2
        mask = mask[start_y:start_y + h, start_x:start_x + w]

        return mask

    def _apply(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            mask_expanded = mask[..., np.newaxis]
        else:
            mask_expanded = mask
        return np.where(mask_expanded > 0.5, data, self.fill_value).astype(data.dtype)
