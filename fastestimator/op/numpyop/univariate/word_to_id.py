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
from fastestimator.util.traceability_util import traceable


@traceable()
class WordtoId(NumpyOp):
    """Converts words to their corresponding id using mapper function or dictionary.

    Args:
        mapping: Mapper function or dictionary
        inputs: Key(s) of sequences to be converted to ids.
        outputs: Key(s) of sequences are converted to ids.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
    """
    def __init__(
            self,
            mapping: Union[Dict[str, int], Callable[[List[str]], List[int]]],
            inputs: Union[str, Iterable[str], Callable],
            outputs: Union[str, Iterable[str]],
            mode: Union[None, str, Iterable[str]] = None,
    ) -> None:
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.in_list, self.out_list = True, True
        assert callable(mapping) or isinstance(mapping, dict), \
            "Incorrect data type provided for `mapping`. Please provide a function or a dictionary."
        self.mapping = mapping

    def forward(self, data: List[List[str]], state: Dict[str, Any]) -> List[np.ndarray]:
        return [self._convert_to_id(elem) for elem in data]

    def _convert_to_id(self, data: List[str]) -> np.ndarray:
        """Flatten the input list and map the token to ids using mapper function or lookup table.

        Args:
            data: Input array of tokens

        Raises:
            Exception: If neither of the mapper function or dictionary object is passed

        Returns:
            Array of token ids
        """
        if callable(self.mapping):
            data = self.mapping(data)
        else:
            data = [self.mapping.get(token) for token in data]
        return np.array(data)
