# Copyright 2021 The FastEstimator Authors. All Rights Reserved.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from skimage.draw import random_shapes
import math

from fastestimator.backend._get_image_dims import get_image_dims
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class RandomShapes(NumpyOp):
    """Add random shapes to an image.

    Args:
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        max_shapes: The maximum number of shapes to add to the image.
        max_size: The maximum size of the shapes to generate.
        intensity_range: The allowable pixel values for the shapes.
        transparency_range: The range of transparency values to be randomly sampled from. 0 means that shapes are
            completely transparent, and 1 means that shapes are completely opaque.

    Raises:
        AssertionError: If the `intensity_range` or `transparency_range` arguments are invalid.
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 max_shapes: int = 3,
                 max_size: Optional[int] = None,
                 intensity_range: Tuple[int, int] = (0, 254),
                 transparency_range: Tuple[float, float] = (0.1, 0.9)):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.max_shapes = max_shapes
        self.max_size = max_size
        assert 0 <= intensity_range[0] <= intensity_range[1] <= 254, "Intensity_range should be in [0, 254]"
        self.intensity_range = intensity_range
        assert 0 <= transparency_range[0] <= transparency_range[1] <= 1, "Transparency_range should be in [0,1]"
        self.transparency_range = transparency_range
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._apply_shapes(elem) for elem in data]

    def _apply_shapes(self, data: np.ndarray) -> np.ndarray:
        channels, height, width = get_image_dims(data)
        # Wrap intensity_range in another tuple for color images. User might have provided nested tuple already though,
        #  so don't wrap those again.
        intensity_range = self.intensity_range
        if channels > 1 and not isinstance(intensity_range[0], tuple):
            intensity_range = (intensity_range, )
        shapes, _ = random_shapes(image_shape=(height, width), max_shapes=self.max_shapes, max_size=self.max_size,
                                  num_channels=channels,
                                  intensity_range=intensity_range,
                                  allow_overlap=True)
        alpha = np.random.uniform(self.transparency_range[0], self.transparency_range[1])

        # Convert shapes to range [0,1] if image is floating point
        normalized_shapes = shapes / 255.0 if np.issubdtype(data.dtype, np.floating) else shapes

        blend = normalized_shapes * alpha + data * (1.0 - alpha)

        # Combine images without whitening non-shape regions
        overlay = np.where(shapes < 255, blend, data)

        return overlay.astype(data.dtype)

    def set_rua_level(self, magnitude_coef: Union[int, float]) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).
        """
        self.max_shapes = math.ceil(magnitude_coef * self.max_shapes)
        self.max_size = None if self.max_size is None else math.ceil(magnitude_coef * self.max_size)

        # Transparency range will keep the same minimum value, but reduce the maximum opacity based on the level.
        transparency_diff = self.transparency_range[1] - self.transparency_range[0]
        transparency_diff = transparency_diff * magnitude_coef
        self.transparency_range = (self.transparency_range[0], self.transparency_range[0] + transparency_diff)
