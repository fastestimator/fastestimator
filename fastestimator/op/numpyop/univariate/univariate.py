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
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np
from albumentations import Compose, ImageOnlyTransform, ReplayCompose

from fastestimator.op.numpyop.numpyop import NumpyOp


class ImageOnlyAlbumentation(NumpyOp):
    """Operators which apply to single images (as opposed to images + masks or images + bounding boxes).

    This is a wrapper for functionality provided by the Albumentations library:
    https://github.com/albumentations-team/albumentations. A useful visualization tool for many of the possible effects
    it provides is available at https://albumentations-demo.herokuapp.com.

    Args:
        func: An Albumentation function to be invoked.
        inputs: Key(s) from which to retrieve data from the data dictionary. If more than one key is provided, the
            `func` will be run in replay mode so that the exact same augmentation is applied to each value.
        outputs: Key(s) under which to write the outputs of this Op back to the data dictionary.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(self,
                 func: ImageOnlyTransform,
                 inputs: Union[str, List[str], Callable],
                 outputs: Union[str, List[str]],
                 mode: Union[None, str, Iterable[str]] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.func = Compose(transforms=[func])
        self.replay_func = ReplayCompose(transforms=[deepcopy(func)])
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        results = [self.replay_func(image=data[0]) if len(data) > 1 else self.func(image=data[0])]
        for i in range(1, len(data)):
            results.append(self.replay_func.replay(results[0]['replay'], image=data[i]))
        return [result["image"] for result in results]
