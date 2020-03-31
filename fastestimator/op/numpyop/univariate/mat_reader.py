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
import os
from typing import Any, Callable, Dict, Iterable, List, Union

from scipy.io import loadmat

from fastestimator.op.numpyop.numpyop import NumpyOp


class MatReader(NumpyOp):
    """Class for reading .mat files
    Args:
        inputs: Key(s) of paths to files to be loaded
        outputs: Key(s) of dictionaries to be output
        mode: What execution mode (train, eval, test, None) to apply this operation
        parent_path: Parent path that will be prepended to a given path
    """
    def __init__(self,
                 inputs: Union[None, str, Iterable[str], Callable] = None,
                 outputs: Union[None, str, Iterable[str]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 parent_path: str = ""):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        if isinstance(self.inputs, List) and isinstance(self.outputs, List):
            assert len(self.inputs) == len(self.outputs), "Input and Output lengths must match"
        self.parent_path = parent_path
        self._loadmat = loadmat

        self.in_list, self.out_list = True, True

    def forward(self, data: List[str], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for path in data:
            path = os.path.normpath(os.path.join(self.parent_path, path))
            img = self._loadmat(path)
            results.append(img)
        return results
