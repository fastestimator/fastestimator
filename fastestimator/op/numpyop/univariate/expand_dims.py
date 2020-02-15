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
from typing import Union, Iterable, Callable, List, Dict, Any

import numpy as np

from fastestimator.op import NumpyOp


class ExpandDims(NumpyOp):
    """Transpose the data (for example to make it channel-width-height instead of width-height-channel)

    Args:
            inputs: Key(s) of array to be expanded
            outputs: Key(s) of array to be expanded
            mode: What execution mode (train, eval, None) to apply this operation
            axis: The axis to expand
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 axis: int = -1):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.axis = axis

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [np.expand_dims(elem, self.axis) for elem in data]
