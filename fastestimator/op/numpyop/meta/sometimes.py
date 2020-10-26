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
from typing import Any, Dict, List

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class Sometimes(NumpyOp):
    """Perform a NumpyOp with a given probability.

    Note that Sometimes should not be used to wrap an op whose output key(s) do not already exist in the data
    dictionary. This would result in a problem when future ops / traces attempt to reference the output key, but
    Sometimes declined to generate it. If you want to create a default value for a new key, simply use a LambdaOp before
    invoking the Sometimes.

    Args:
        numpy_op: The operator to be performed.
        prob: The probability of execution, which should be in the range: [0-1).
    """
    def __init__(self, numpy_op: NumpyOp, prob: float = 0.5) -> None:
        # We're going to try to collect any missing output keys from the data dictionary so that they don't get
        # overridden when Sometimes chooses not to execute.
        inps = set(numpy_op.inputs)
        outs = set(numpy_op.outputs)
        self.extra_inputs = list(outs - inps)  # Used by traceability
        self.inp_idx = len(numpy_op.inputs)
        super().__init__(inputs=numpy_op.inputs + self.extra_inputs, outputs=numpy_op.outputs, mode=numpy_op.mode)
        # Note that in_list and out_list will always be true
        self.op = numpy_op
        self.prob = prob

    def __getstate__(self) -> Dict[str, Dict[Any, Any]]:
        return {'op': self.op.__getstate__() if hasattr(self.op, '__getstate__') else {}}

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        """Execute the wrapped operator a certain fraction of the time.

        Args:
            data: The information to be passed to the wrapped operator.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The original `data`, or the `data` after running it through the wrapped operator.
        """
        if self.prob > np.random.uniform():
            data = data[:self.inp_idx]  # Cut off the unnecessary inputs
            if not self.op.in_list:
                data = data[0]
            data = self.op.forward(data, state)
            if not self.op.out_list:
                data = [data]
        else:
            data = [data[self.inputs.index(out)] for out in self.outputs]
        return data
