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
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np

from fastestimator.op.numpyop.numpyop import NumpyOp


class Tokenize(NumpyOp):

    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 tokenize_fn: Union[None, Callable] = None,
                 do_lower_case: bool = False):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True
        self.tokenize_fn = tokenize_fn
        self.do_lower_case = do_lower_case

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        if self.tokenize_fn:
            return [self.tokenize_fn(seq) for seq in data]
        return [self._apply_tokenization(seq) for seq in data]

    def _apply_tokenization(self, data):
        data = data.split()
        if self.do_lower_case:
            return data.lower()
        return data
