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

import cv2
import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class SimulateLowResolution(NumpyOp):
    """Simulate lower resolution images by downscaling and then upscaling.

    This augmentation is useful for training models that need to be robust to variable scan quality, which is common
    in medical imaging where different scanners, protocols, and reconstruction parameters produce images at varying
    effective resolutions. The image is first downscaled by a random factor and then upscaled back to the original
    size.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        zoom_range: The range of zoom factors. Values < 1.0 will simulate lower resolution. For example,
            (0.5, 1.0) means the image will be downscaled to between 50% and 100% of its original size.
        interpolation_down: Interpolation method for downscaling. One of cv2.INTER_NEAREST, cv2.INTER_LINEAR,
            cv2.INTER_AREA, cv2.INTER_CUBIC.
        interpolation_up: Interpolation method for upscaling back to original size.

    Image types:
        uint8, float32
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 zoom_range: Tuple[float, float] = (0.5, 1.0),
                 interpolation_down: int = cv2.INTER_AREA,
                 interpolation_up: int = cv2.INTER_LINEAR,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        assert 0 < zoom_range[0] <= zoom_range[1] <= 1.0, \
            "zoom_range values must be in (0, 1] with zoom_range[0] <= zoom_range[1]"
        self.zoom_range = zoom_range
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        if zoom >= 1.0:
            return data
        return [self._apply(elem, zoom) for elem in data]

    def _apply(self, data: np.ndarray, zoom: float) -> np.ndarray:
        orig_shape = data.shape[:2]
        small_h = max(1, int(round(orig_shape[0] * zoom)))
        small_w = max(1, int(round(orig_shape[1] * zoom)))
        downscaled = cv2.resize(data, (small_w, small_h), interpolation=self.interpolation_down)
        upscaled = cv2.resize(downscaled, (orig_shape[1], orig_shape[0]), interpolation=self.interpolation_up)
        # Preserve channel dimension if it was squeezed by cv2
        if data.ndim == 3 and upscaled.ndim == 2:
            upscaled = upscaled[..., np.newaxis]
        return upscaled.astype(data.dtype)
