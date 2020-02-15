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
import random
from typing import Dict, Any, List

import numpy as np

from fastestimator.op import NumpyOp


class OneOf(NumpyOp):
    """Perform one of several possible NumpyOps

        Args:
            numpy_ops: A list of ops to choose between
    """
    def __init__(self, *numpy_ops: NumpyOp):
        inputs = numpy_ops[0].inputs
        outputs = numpy_ops[0].outputs
        mode = numpy_ops[0].mode
        for op in numpy_ops[1:]:
            assert inputs == op.inputs, "All ops within a OneOf must share the same inputs"
            assert outputs == op.outputs, "All ops within a OneOf must share the same outputs"
            assert mode == op.mode, "All ops within a OneOf must share the same mode"
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.numpy_ops = numpy_ops

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        """Execute the operator with probability

        Args:
            data: Tensor to be resized.
            state: Information about the current execution context.

        Returns:
            output tensor.
        """
        return random.choice(self.numpy_ops).forward(data, state)
