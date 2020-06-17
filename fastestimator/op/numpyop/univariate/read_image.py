# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
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
import os
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class ReadImage(NumpyOp):
    """A class for reading png or jpg images from disk.

    Args:
        inputs: Key(s) of paths to images to be loaded.
        outputs: Key(s) of images to be output.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        parent_path: Parent path that will be prepended to a given path.
        color_flag: Whether to read the image as 'color', 'grey', or one of the cv2.IMREAD flags.

    Raises:
        AssertionError: If `inputs` and `outputs` have mismatched lengths, or the `color_flag` is unacceptable.
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 parent_path: str = "",
                 color_flag: Union[str, int] = cv2.IMREAD_COLOR):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(self.inputs, List) and isinstance(self.outputs, List):
            assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.parent_path = parent_path
        assert isinstance(color_flag, int) or color_flag in {'color', 'gray', 'grey'}, \
            f"Unacceptable color_flag value: {color_flag}"
        self.color_flag = color_flag
        if self.color_flag == "color":
            self.color_flag = cv2.IMREAD_COLOR
        elif self.color_flag in {"gray", "grey"}:
            self.color_flag = cv2.IMREAD_GRAYSCALE
        self.in_list, self.out_list = True, True

    def forward(self, data: List[str], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._read(elem) for elem in data]

    def _read(self, path: str) -> np.ndarray:
        path = os.path.normpath(os.path.join(self.parent_path, path))
        img = cv2.imread(path, self.color_flag)
        if self.color_flag in {
                cv2.IMREAD_COLOR, cv2.IMREAD_REDUCED_COLOR_2, cv2.IMREAD_REDUCED_COLOR_4, cv2.IMREAD_REDUCED_COLOR_8
        }:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(img, np.ndarray):
            raise ValueError('cv2 did not read correctly for file "{}"'.format(path))
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        return img
