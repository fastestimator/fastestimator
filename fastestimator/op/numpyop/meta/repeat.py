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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp, forward_numpyop
from fastestimator.util.traceability_util import traceable


@traceable()
class Repeat(NumpyOp):
    """Repeat a NumpyOp several times in a row.

    Args:
        op: A NumpyOp to be run one or more times in a row.
        repeat: How many times to repeat the `op`. This can also be a function return, in which case the function input
            names will be matched to keys in the data dictionary, and the `op` will be repeated until the function
            evaluates to False. The function evaluation will happen at the end of a forward call, so the `op` will
            always be evaluated at least once.
        max_iter: A limit to how many iterations will be run (or None for no limit).

    Raises:
        ValueError: If `repeat`, `op`, or max_iter are invalid.
    """
    def __init__(self, op: NumpyOp, repeat: Union[int, Callable[..., bool]] = 1,
                 max_iter: Optional[int] = None) -> None:
        self.repeat_inputs = []
        extra_reqs = []
        if max_iter is None:
            self.max_iter = max_iter
        else:
            if max_iter < 1:
                raise ValueError(f"Repeat requires max_iter to be >=1, but got {max_iter}")
            self.max_iter = max_iter - 1  # -1 b/c the first invocation happens outside the while loop
        if isinstance(repeat, int):
            if repeat < 1:
                raise ValueError(f"Repeat requires repeat to be >= 1, but got {repeat}")
            if max_iter:
                raise ValueError("Do not set max_iter when repeat is an integer")
        else:
            self.repeat_inputs.extend(inspect.signature(repeat).parameters.keys())
            extra_reqs = list(set(self.repeat_inputs) - set(op.outputs))
        self.repeat = repeat
        super().__init__(inputs=op.inputs + extra_reqs, outputs=op.outputs, mode=op.mode, ds_id=op.ds_id)
        self.ops = [op]

    @property
    def op(self) -> NumpyOp:
        return self.ops[0]

    def __getstate__(self) -> Dict[str, List[Dict[Any, Any]]]:
        return {'ops': [elem.__getstate__() if hasattr(elem, '__getstate__') else {} for elem in self.ops]}

    def set_rua_level(self, magnitude_coef: float) -> None:
        """Set the augmentation intensity based on the magnitude_coef.

        This method is specifically designed to be invoked by the RUA Op.

        Args:
            magnitude_coef: The desired augmentation intensity (range [0-1]).

        Raises:
            AttributeError: If the 'op' doesn't have a 'set_rua_level' method.
        """
        if hasattr(self.op, "set_rua_level") and inspect.ismethod(getattr(self.op, "set_rua_level")):
            self.op.set_rua_level(magnitude_coef=magnitude_coef)
        else:
            raise AttributeError(
                "RUA Augmentations should have a 'set_rua_level' method but it's not present in Op: {}".format(
                    self.op.__class__.__name__))

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        data = {key: elem for key, elem in zip(self.inputs, data)}
        if isinstance(self.repeat, int):
            for i in range(self.repeat):
                forward_numpyop(self.ops, data, state)
        else:
            forward_numpyop(self.ops, data, state)
            i = 0
            while self.repeat(*[data[var_name] for var_name in self.repeat_inputs]):
                if self.max_iter and i >= self.max_iter:
                    break
                forward_numpyop(self.ops, data, state)
                i += 1
        return [data[key] for key in self.outputs]
