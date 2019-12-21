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
from fastestimator.op import NumpyOp
import numpy as np


class Probability(NumpyOp):
    """Perform a NumpyOp with a given probability

        Args:
            numpy_op (NumpyOp): The target op instance
            prob (float): The probability of execution [0-1)
    """
    def __init__(self, numpy_op, prob=0.5):
        super().__init__(inputs=numpy_op.inputs, outputs=numpy_op.outputs, mode=numpy_op.mode)
        self.numpy_op = numpy_op
        self.prob = prob

    def forward(self, data, state):
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
