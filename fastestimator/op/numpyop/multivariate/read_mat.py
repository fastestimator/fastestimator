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
from typing import Any, Dict, Iterable, List, Union

from scipy.io import loadmat

from fastestimator.op.numpyop.numpyop import NumpyOp
from fastestimator.util.util import to_list


class ReadMat(NumpyOp):
    """Class for reading .mat files, it works when every sample has a separate .mat file.
    Args:
        file_in: Key of file path to be loaded.
        keys_in: Key(s) of dictionaries to read.
        keys_out: Keys(s) of dictionaries to output.
        mode: What execution mode (train, eval, test, None) to apply this operation
        parent_path: Parent path that will be prepended to a given filepath
    """
    def __init__(self,
                 file_in: str,
                 keys_in: Union[str, Iterable[str]],
                 keys_out: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 parent_path: str = ""):
        self.file_in = file_in
        self.keys_in = to_list(keys_in)
        self.keys_out = to_list(keys_out)
        self.parent_path = parent_path
        assert len(self.keys_in) == len(self.keys_out), "keys_in and keys_out lengths must match"
        super().__init__(inputs=self.file_in, outputs=self.keys_out, mode=mode)

    def forward(self, data: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = loadmat(os.path.normpath(os.path.join(self.parent_path, data)))
        results = [data[key] for key in self.keys_in]
        return results
