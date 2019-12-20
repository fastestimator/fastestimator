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
from typing import Optional, List, Tuple, Union
from copy import deepcopy

from albumentations import ImageOnlyTransform, ReplayCompose, Compose

from fastestimator.op import NumpyOp


class ImageOnlyAlbumentation(NumpyOp):
    def __init__(self,
                 func: ImageOnlyTransform,
                 inputs: Union[List[str], str, None] = None,
                 outputs: Union[List[str], str, None] = None,
                 mode: Optional[str] = None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(inputs, List) and isinstance(outputs, List):
            assert len(inputs) == len(outputs), "Input and Output lengths must match"
        self.func = Compose(transforms=[func])
        self.replay_func = ReplayCompose(transforms=[deepcopy(func)])

    def forward(self, data, state):
        if isinstance(data, (List, Tuple)):
            results = [self.replay_func(image=data[0])]
            for i in range(1, len(data)):
                results.append(self.replay_func.replay(results[0]['replay'], image=data[i]))
            return [result["image"] for result in results]
        else:
            return self.func(image=data)["image"]
