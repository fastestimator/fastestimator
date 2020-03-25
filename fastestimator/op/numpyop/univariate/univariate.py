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
from typing import Any, Dict, List, Optional, Union

import numpy as np

from albumentations import Compose, ImageOnlyTransform, ReplayCompose
from fastestimator.op.numpyop.numpyop import NumpyOp


class ImageOnlyAlbumentation(NumpyOp):
    def __init__(self,
                 func: ImageOnlyTransform,
                 inputs: Union[List[str], str, None] = None,
                 outputs: Union[List[str], str, None] = None,
                 mode: Optional[str] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(self.inputs, List) and isinstance(self.outputs, List):
            assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.func = Compose(transforms=[func])
        self.replay_func = ReplayCompose(transforms=[deepcopy(func)])
        self.in_list, self.out_list = True, True

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        results = [self.replay_func(image=data[0]) if len(data) > 1 else self.func(image=data[0])]
        for i in range(1, len(data)):
            results.append(self.replay_func.replay(results[0]['replay'], image=data[i]))
        return [result["image"] for result in results]
