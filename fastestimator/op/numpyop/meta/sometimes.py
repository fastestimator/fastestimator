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
from typing import Dict, Any, List, Union

import numpy as np

from fastestimator.op import NumpyOp


class Sometimes(NumpyOp):
    """Perform a NumpyOp with a given probability

        Args:
            numpy_op: The target op instance
            prob: The probability of execution [0-1)
    """
    def __init__(self, numpy_op: NumpyOp, prob: float = 0.5):
        super().__init__(inputs=numpy_op.inputs, outputs=numpy_op.outputs, mode=numpy_op.mode)
        self.numpy_op = numpy_op
        self.prob = prob

    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        """Execute the operator with probability

        Args:
            data: Tensor to be resized.
            state: Information about the current execution context.

        Returns:
            output tensor.
        """
        if self.prob > np.random.uniform():
            data = self.numpy_op.forward(data, state)
        return data
