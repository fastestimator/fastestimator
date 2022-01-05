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
import inspect
import random
from typing import Any, Dict, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import Batch, NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class OneOf(NumpyOp):
    """Perform one of several possible NumpyOps.

    Args:
        *numpy_ops: A list of ops to choose between with uniform probability.
    """
    def __init__(self, *numpy_ops: NumpyOp) -> None:
        inputs = numpy_ops[0].inputs
        outputs = numpy_ops[0].outputs
        mode = numpy_ops[0].mode
        ds_id = numpy_ops[0].ds_id
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.in_list = numpy_ops[0].in_list
        self.out_list = numpy_ops[0].out_list
        for op in numpy_ops[1:]:
            assert not isinstance(op, Batch), "Cannot nest the Batch op inside OneOf"
            assert inputs == op.inputs, "All ops within a OneOf must share the same inputs"
            assert self.in_list == op.in_list, "All ops within OneOf must share the same input configuration"
            assert outputs == op.outputs, "All ops within a OneOf must share the same outputs"
            assert self.out_list == op.out_list, "All ops within OneOf must share the same output configuration"
            assert mode == op.mode, "All ops within a OneOf must share the same mode"
            assert ds_id == op.ds_id, "All ops within a OneOf must share the same ds_id"
        self.ops = numpy_ops

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'ops': [elem.__getstate__() if hasattr(elem, '__getstate__') else {} for elem in self.ops]}

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).

        Raises:
            AttributeError: If ops don't have a 'set_rua_level' method.
        """
        for op in self.ops:
            if hasattr(op, "set_rua_level") and inspect.ismethod(getattr(op, "set_rua_level")):
                op.set_rua_level(magnitude_coef=magnitude_coef)
            else:
                raise AttributeError(
                    "RUA Augmentations should have a 'set_rua_level' method but it's not present in Op: {}".format(
                        op.__class__.__name__))

    def forward(self, data: Union[np.ndarray, List[np.ndarray]],
                state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        """Execute a randomly selected op from the list of `numpy_ops`.

        Args:
            data: The information to be passed to one of the wrapped operators.
            state: Information about the current execution context, for example {"mode": "train"}.

        Returns:
            The `data` after application of one of the available numpyOps.
        """
        return random.choice(self.ops).forward(data, state)

    def forward_batch(self,
                      data: Union[np.ndarray, List[np.ndarray]],
                      state: Dict[str, Any]) -> Union[np.ndarray, List[np.ndarray]]:
        return random.choice(self.ops).forward_batch(data, state)
