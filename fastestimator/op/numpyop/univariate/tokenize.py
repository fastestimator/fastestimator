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
    """Split the sequences into tokens.

    Tokenize split the document/sequence into tokens and at the same time perform additional operations on tokens if
    defined in the passed function object. By default, tokenize only splits the sequences into tokens.

    Args:
        inputs: Key(s) of sequences to be tokenized.
        outputs: Key(s) of sequences that are tokenized.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        tokenize_fn: Tokenization function object.
        do_lower_case: Whether to convert tokens to lowercase.
    """

    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 tokenize_fn: Union[None, Callable] = None,
                 do_lower_case: bool = False) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True
        self.tokenize_fn = tokenize_fn
        self.do_lower_case = do_lower_case

    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        if self.tokenize_fn:
            return [self.tokenize_fn(seq) for seq in data]
        return [self._apply_tokenization(seq) for seq in data]

    def _apply_tokenization(self, data: np.ndarray) -> List[str]:
        """Split the sequence into tokens and apply lowercase if `do_lower_case` is set.

        Args:
            data: Input sequence

        Returns:
            List of tokens
        """
        data = data.split()
        if self.do_lower_case:
            return data.lower()
        return data
