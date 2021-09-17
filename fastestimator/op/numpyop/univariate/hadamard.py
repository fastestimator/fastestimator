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
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from scipy.linalg import hadamard

from fastestimator.backend.gather import gather
from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.traceability_util import traceable


@traceable()
class Hadamard(NumpyOp):
    """Convert integer labels into hadamard code representations.

    Args:
        inputs: Key of the input tensor(s) to be converted.
        outputs: Key of the output tensor(s) in hadamard code representation.
        n_classes: How many classes are there in the inputs.
        code_length: What code length to use. Will default to the smallest power of 2 which is >= the number of classes.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".

    Raises:
        ValueError: If an unequal number of `inputs` and `outputs` are provided, or if `code_length` is invalid.
    """
    def __init__(self,
                 inputs: Union[str, List[str]],
                 outputs: Union[str, List[str]],
                 n_classes: int,
                 code_length: Optional[int] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, ds_id=ds_id)
        self.in_list, self.out_list = True, True
        if len(self.inputs) != len(self.outputs):
            raise ValueError("Hadamard requires the same number of input and output keys.")
        self.n_classes = n_classes
        if code_length is None:
            code_length = 1 << (n_classes - 1).bit_length()
        if code_length <= 0 or (code_length & (code_length - 1) != 0):
            raise ValueError(f"code_length must be a positive power of 2, but got {code_length}.")
        if code_length < n_classes:
            raise ValueError(f"code_length must be >= n_classes, but got {code_length} and {n_classes}.")
        self.code_length = code_length
        labels = hadamard(self.code_length).astype(np.float32)
        labels[np.arange(0, self.code_length, 2), 0] = -1  # Make first column alternate
        labels = labels[:self.n_classes]
        self.labels = labels

    def forward(self, data: List[Union[int, np.ndarray]], state: Dict[str, Any]) -> List[np.ndarray]:
        # TODO - also support one hot with smoothed labels?
        return [gather(tensor=self.labels, indices=np.array(inp)) for inp in data]
