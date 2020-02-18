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
from typing import Union, Iterable, Callable, Any, Dict, List

import cv2
import numpy as np

from fastestimator.op import NumpyOp


class ReadImage(NumpyOp):
    """Class for reading png or jpg images
    Args:
        inputs: Key(s) of paths to images to be loaded
        outputs: Key(s) of images to be output
        mode: What execution mode (train, eval, None) to apply this operation
        parent_path: Parent path that will be prepended to a given path
        grey_scale: Whether or not to read the image as grayscale
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 parent_path: str = "",
                 grey_scale: bool = False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(self.inputs, List) and isinstance(self.outputs, List):
            assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.parent_path = parent_path
        self.color_flag = cv2.IMREAD_COLOR
        self.grey_scale = grey_scale
        if grey_scale:
            self.color_flag = cv2.IMREAD_GRAYSCALE
        self.in_list, self.out_list = True, True

    def forward(self, data: List[str], state: Dict[str, Any]) -> List[np.ndarray]:
        results = []
        for path in data:
            path = os.path.normpath(os.path.join(self.parent_path, path))
            img = cv2.imread(path, self.color_flag)
            if not self.grey_scale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                raise ValueError('cv2 did not read correctly for file "{}"'.format(path))
            if self.grey_scale:
                img = np.expand_dims(img, -1)
            results.append(img)
        return results
