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
import collections
import types
from typing import Any, Callable, Dict, Iterable, List, Union

import numpy as np
from fastestimator.op.numpyop.numpyop import NumpyOp


class WordtoId(NumpyOp):
    """Converts words to their corresponding id using mapper function or dictionary.

    Args:
        mapping: Mapper function or dictionary
        inputs: Key(s) of sequences to be converted to ids.
        outputs: Key(s) of sequences are converted to ids.
        mode: What execution mode (train, eval, None) to apply this operation.
    """
    def __init__(self,
                 mapping: Union[None, dict, Callable],
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True
        self.mapping = mapping


    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._convert_to_id(elem) for elem in data]

    def _convert_to_id(self, data: np.ndarray) -> np.ndarray:
        """Flatten the input list and map the token to ids using mapper function or lookup table.

        Args:
            data: Input array of tokens

        Raises:
            Exception: If neither of the mapper function or dictionary object is passed

        Returns:
            Array of token ids
        """
        data = self._flatten_list(data)
        if callable(self.mapping):
            data = self.mapping(data)
        elif isinstance(self.mapping, dict):
            data = [self.mapping.get(token) for token in data]
        else:
            raise Exception('Must pass a function type or dictionary object for mapping')
        return np.array(data)

    def _flatten_list(self, data: List[List[str]]) -> str:
        """Flatten the nested list

        Args:
            data: Nested list of tokens

        Yields:
            Token string
        """
        for el in data:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from self._flatten_list(el)
            else:
                yield el
